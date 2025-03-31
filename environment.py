import os

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import logging
import pytz
import indicators
import numpy as np

est = pytz.timezone('America/New_York')
utc = pytz.utc

API = "api-fxpractice.oanda.com"
STREAM_API = "stream-fxpractice.oanda.com"

ACCOUNT_ID = "101-002-25776676-003"
TOKEN = "d8e783a23ff8bab21476e440b3d578ef-207d5f7676d7d82839a50d4907b3d6e6"


class EnviroBatchProcess:
    def __init__(self, instrument, train_start, train_end, batch_size, indicator_select, testing=False):
        self.instrument = instrument
        self.train_start = datetime.strptime(train_start, "%Y-%m-%d")
        self.train_end = datetime.strptime(train_end, "%Y-%m-%d")
        self.batch_size = batch_size
        self.indicator_select = indicator_select

        self.path_to_data = './data/test_data/full_training_data_' if testing else './data/full_training_data_'
        self.path_to_indicator = './indicators/test/' if testing else './indicators/train/'

        self.start_opening = 0  # this is opening price for compression and decompression
        self.first_date = datetime.now()  # just as a placeholder for opening date for compression and decompression

        self.orders = {"open": [], "closed": []}
        self.year_time_step = 60  # keeps track of which index in the year currently on
        self.balance = 0
        self.done = False
        self.commission = 0.5

        self.year_data_shape = ()
        self.year_data_filename = 'train_year_data.dat' if not testing else 'test_year_data.dat'
        # self.fetch_current_year_data()
        if not os.path.exists(self.year_data_filename):
            self.fetch_all_years_data()
        else:
            self.year_data_shape = (4777984, 5) if not testing else (2580736, 5)
        self.indicator_class = indicators.BatchIndicators(self.year_data_filename, self.year_data_shape,
                                                          indicator_select, testing=testing)
        self.env_out = self.get_env_state()

    def fetch_current_year_data(self, year=None):
        self.current_year = year if year else self.current_year
        with open(self.path_to_data + self.current_year + '.json', 'r') as f:
            # data is of json type because iterating through it all is not that slow.
            data = json.load(f)

        self.first_date = datetime.strptime(data[0][-1][:19], '%Y-%m-%d %H:%M:%S')
        uncompressed_data = []

        for candle in data:
            candle_to_append = self.decompress(candle)
            if len(uncompressed_data) == 0:
                uncompressed_data.append(candle_to_append)
                continue
            second_difference_to_previous = candle_to_append[-1] - uncompressed_data[-1][-1]
            minute_difference_to_previous = (second_difference_to_previous.total_seconds() / 60) - 1
            duplicate_candle = []
            for i in range(int(minute_difference_to_previous)):
                hold = uncompressed_data[-1].copy()
                hold[-1] = hold[-1] + timedelta(minutes=i+1)
                duplicate_candle.append(hold)
            uncompressed_data.extend(duplicate_candle)
            uncompressed_data.append(candle_to_append)

        uncompressed_data = np.array(uncompressed_data)
        self.year_data_shape = uncompressed_data.shape
        arr = np.memmap(self.year_data_filename, dtype=object, mode='w+', shape=self.year_data_shape)
        for i in range(self.year_data_shape[0]):
            arr[i] = uncompressed_data[i]
        arr.flush()

    def fetch_all_years_data(self):
        uncompressed_data = []
        for year in range(self.train_start.year, self.train_end.year + 1):
            with open(self.path_to_data + str(year) + '.json', 'r') as f:
                data = json.load(f)

            self.first_date = datetime.strptime(data[0][-1][:19], '%Y-%m-%d %H:%M:%S')

            for candle in data:
                candle_to_append = self.decompress(candle)
                if len(uncompressed_data) == 0:
                    if candle_to_append[-1] == self.train_start.timestamp():
                        uncompressed_data.append(candle_to_append)
                    else:
                        difference_to_start = ((candle_to_append[-1] - self.train_start.timestamp()) / 60) - 1
                        start_candle = candle_to_append.copy()
                        start_candle[-1] = self.train_start.timestamp()
                        beginning_candles = [start_candle]
                        for i in range(int(difference_to_start)):
                            hold = start_candle.copy()
                            hold[-1] += ((i+1)*60)
                            beginning_candles.append(hold)
                        uncompressed_data.extend(beginning_candles)
                        uncompressed_data.append(candle_to_append)
                    continue
                minute_difference_to_previous = ((candle_to_append[-1] - uncompressed_data[-1][-1]) / 60) - 1
                duplicate_candle = []
                for i in range(int(minute_difference_to_previous)):
                    hold = uncompressed_data[-1].copy()
                    hold[-1] += ((i+1)*60)
                    duplicate_candle.append(hold)
                uncompressed_data.extend(duplicate_candle)
                uncompressed_data.append(candle_to_append)
        padding_required = self.batch_size - (len(uncompressed_data) % self.batch_size)
        # print(len(uncompressed_data), padding_required)
        padding = [[0, 0, 0, 0, 0]] * padding_required
        uncompressed_data.extend(padding)
        uncompressed_data = np.array(uncompressed_data)
        print(uncompressed_data.shape)
        self.year_data_shape = uncompressed_data.shape

        arr = np.memmap(self.year_data_filename, dtype=np.float64, mode='w+', shape=self.year_data_shape)
        for i in range(self.year_data_shape[0]):
            arr[i] = uncompressed_data[i]
        arr.flush()

    def decompress(self, candle):
        # for Open, High, Low, Close, Date
        ret_candle = [0, 0, 0, 0, '']
        try:
            multi_start_date = datetime.strptime(candle[-1][:19], '%Y-%m-%d %H:%M:%S')  # this will throw the error
            # if we make it past this line it means that this is a new 6-hour block of data being provided so update
            # both the date and the opening price

            # separate it into 2 variables to make sure that if error is thrown self.first_date isn't changed
            self.first_date = multi_start_date
            ret_candle[-1] = self.first_date.timestamp()

            self.start_opening = candle[0]
            ret_candle[0] = self.start_opening
            ret_candle[1] = round(self.start_opening - candle[1], 2)  # x = start - current => current = start - x
            ret_candle[2] = round(self.start_opening - candle[2], 2)
            ret_candle[3] = round(self.start_opening - candle[3], 2)

        except ValueError:
            ret_candle[-1] = (self.first_date - timedelta(minutes=int(candle[-1]))).timestamp()
            ret_candle[0] = round(self.start_opening - candle[0], 2)
            ret_candle[1] = round(self.start_opening - candle[1], 2)
            ret_candle[2] = round(self.start_opening - candle[2], 2)
            ret_candle[3] = round(self.start_opening - candle[3], 2)

        return ret_candle

    def get_env_state(self):
        returning_batch = []
        if self.year_time_step + self.batch_size > self.year_data_shape[0]:
            self.done = True
            return np.array([])
        end_index = self.year_time_step + self.batch_size
        year_data = np.memmap(self.year_data_filename, dtype=np.float64, mode='r', shape=self.year_data_shape)

        year_indicators = []
        file_map = [self.path_to_indicator+'rsi_data.dat', self.path_to_indicator+'mac_data.dat',
                    self.path_to_indicator+'ob_data.dat', self.path_to_indicator+'fvg_data.dat',
                    self.path_to_indicator+'news_data.dat']
        shape_map = [(self.year_data_shape[0],), (self.year_data_shape[0],), (self.year_data_shape[0], 10),
                     (self.year_data_shape[0], 20), (self.year_data_shape[0],)]
        for flag_index, flag in enumerate(self.indicator_select):
            if flag:
                year_indicators.append(np.memmap(file_map[flag_index], dtype=np.float64, mode='r', shape=shape_map[flag_index]))

        for i in range(self.year_time_step, end_index):
            individual_batch = np.array(year_data[i-60:i])
            # individual_batch[:, 4] = [dt.timestamp() for dt in individual_batch[:, 4]]

            for j in year_indicators:
                individual_batch = np.column_stack((individual_batch, np.array(j[i-60:i])))
            returning_batch.append(individual_batch)
        return np.array(returning_batch, dtype=np.float64)

    # actions is a list of 256
    def step(self, actions):
        # fetch the data for the next year
        year_data = np.memmap(self.year_data_filename, dtype=np.float64, mode='r', shape=self.year_data_shape)
        if self.year_time_step >= self.year_data_shape[0]:
            self.done = True

            # once after running through the entire year's data will write all the orders to a file
            with open('orders.json', 'w') as write:
                json.dump(self.orders, write)

        reward_multiplier = {
            # 'Micro E-mini NASDAQ': 2,
            # 'E-mini NASDAQ': 20,
            # 'Micro E-mini S&P': 5,
            # 'E-mini S&P': 50,
            'SPX500_USD': 5,
            'NAS100_USD': 2
        }

        returning_reward = []  # should be the unrealized profit
        returning_realized = []  #should be the realized profit
        # calculating the realized reward
        # print(actions)
        for action_index, action in enumerate(actions):
            if action != 'hold':
                if len(self.orders['open']) == 0:
                    self.orders['open'].append({
                        'entry_datetime': year_data[self.year_time_step+action_index][-1],
                        'exit_datetime': '',
                        'entry_price': year_data[self.year_time_step+action_index][-2],
                        'exit_price': '',
                        'order': action
                    })
                    self.balance -= self.commission  # commission for now is 1
                    returning_reward.append(0)
                else:
                    if (action == 'buy' and self.orders['open'][0]['order'] == 'sell') or (action == 'sell' and self.orders['open'][0]['order'] == 'buy'):
                        move_to_close = self.orders['open'].pop()
                        move_to_close['exit_datetime'] = year_data[self.year_time_step+action_index][-1]
                        move_to_close['exit_price'] = year_data[self.year_time_step+action_index][-2]
                        self.orders['closed'].append(move_to_close)
                        # this reward below just calculates the difference between entry and exit apply a commission
                        # rate of 1 (for now)
                        reward = ((move_to_close['entry_price'] - move_to_close['exit_price']) *
                                  reward_multiplier[self.instrument]) if action == 'buy' else (
                                ((move_to_close['exit_price'] - move_to_close['entry_price']) *
                                 reward_multiplier[self.instrument]))
                        self.balance += round(reward - self.commission, 2)
                        returning_reward.append(0)
                    else:
                        current_close_price = year_data[self.year_time_step + action_index][-2]
                        opening_price = self.orders['open'][0]['entry_price']
                        reward = ((opening_price - current_close_price) *
                                  reward_multiplier[self.instrument]) if action == 'sell' else (
                            ((current_close_price - opening_price) *
                             reward_multiplier[self.instrument]))
                        # self.balance -= self.commission
                        returning_reward.append(round(reward, 2))
            else:
                if len(self.orders['open']) == 0:
                    returning_reward.append(0)
                else:
                    current_close_price = year_data[self.year_time_step + action_index][-2]
                    opening_price = self.orders['open'][0]['entry_price']
                    previous_action = self.orders['open'][0]['order']
                    reward = ((opening_price - current_close_price) *
                              reward_multiplier[self.instrument]) if previous_action == 'sell' else (
                        ((current_close_price - opening_price) *
                         reward_multiplier[self.instrument]))
                    returning_reward.append(round(reward, 2))

            # just hard coded for now 100 which is 2% or 50 handles (max draw down)
            if returning_reward[-1] < -100 and len(self.orders['open']) != 0:
                move_to_close = self.orders['open'].pop()
                move_to_close['exit_datetime'] = year_data[self.year_time_step + action_index][-1]
                move_to_close['exit_price'] = year_data[self.year_time_step + action_index][-2]
                self.orders['closed'].append(move_to_close)
                self.balance += returning_reward[-1] - self.commission
                returning_reward[-1] = 0

            # max time limit, trades held at 6pm est will be auto liquidated
            if (datetime.fromtimestamp(int(year_data[self.year_time_step + action_index][-1])).hour == 18 and
                    datetime.fromtimestamp(int(year_data[self.year_time_step + action_index][-1])).minute == 0 and
                    len(self.orders['open']) != 0):
                move_to_close = self.orders['open'].pop()
                move_to_close['exit_datetime'] = year_data[self.year_time_step + action_index][-1]
                move_to_close['exit_price'] = year_data[self.year_time_step + action_index][-2]
                self.orders['closed'].append(move_to_close)
                self.balance += returning_reward[-1] - self.commission
                returning_reward[-1] = 0

            returning_realized.append(self.balance)

        # print(returning_reward)
        # updating state space to get next batch ie [:256] => [256:512]
        self.year_time_step += self.batch_size
        self.env_out = self.get_env_state()

        return [self.env_out, returning_reward, returning_realized]


if __name__ == '__main__':
    train_start = '2011-01-03'
    train_end = '2020-02-03'
    # # train_start = datetime.strptime(train_start, "%Y-%m-%d %H:%M:%S")
    # # train_final_end = datetime.strptime(train_end, "%Y-%m-%d %H:%M:%S")
    # # train_end = train_start
    instrument = 'NAS100_USD'
    a = EnviroBatchProcess(instrument, train_start, train_end, 256)
    print(a.env_out.shape)
    # for i in a.year_indicators[0]:
    #     print(i)
    # for i in a.year_data:
    #     print(i)
    # 2011-12-30 16:14:00, Current: 2012-01-03 06:00:00
    # print(a.fetch_current_year_data(2020))
    # print(a.year_data)
