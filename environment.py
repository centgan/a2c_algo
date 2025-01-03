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
    def __init__(self, instrument, train_start, train_end, batch_size):
        self.instrument = instrument
        self.train_start = datetime.strptime(train_start, "%Y-%m-%d")
        self.train_end = datetime.strptime(train_end, "%Y-%m-%d")
        self.batch_size = batch_size

        self.current_year = self.train_start.year

        self.start_opening = 0  # this is opening price for compression and decompression
        self.first_date = datetime.now()  # just as a placeholder for opening date for compression and decompression

        self.orders = {"open": [], "closed": []}
        self.year_time_step = 60  # keeps track of which index in the year currently on
        self.done = False

        self.year_data_shape = ()
        self.year_data_filename = 'year_data.dat'
        # self.fetch_current_year_data()
        self.fetch_all_years_data()
        self.indicator_class = indicators.BatchIndicators(self.year_data_filename, self.year_data_shape, rsi_flag=True, mac_flag=True)
        self.env_out = self.get_env_state()

    def fetch_current_year_data(self, year=None):
        self.current_year = year if year else self.current_year
        with open(f'./data/full_training_data_{self.current_year}.json', 'r') as f:
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
            with open(f'./data/full_training_data_{year}.json', 'r') as f:
                data = json.load(f)

            self.first_date = datetime.strptime(data[0][-1][:19], '%Y-%m-%d %H:%M:%S')

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
                    hold[-1] = hold[-1] + timedelta(minutes=i + 1)
                    duplicate_candle.append(hold)
                uncompressed_data.extend(duplicate_candle)
                uncompressed_data.append(candle_to_append)
        padding_required = 256 - (len(uncompressed_data) % 256)
        # print(len(uncompressed_data), padding_required)
        padding = [[0, 0, 0, 0, datetime.now()]] * padding_required
        uncompressed_data.extend(padding)
        uncompressed_data = np.array(uncompressed_data)
        self.year_data_shape = uncompressed_data.shape
        print(self.year_data_shape)
        arr = np.memmap(self.year_data_filename, dtype=object, mode='w+', shape=self.year_data_shape)
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
            ret_candle[-1] = self.first_date

            self.start_opening = candle[0]
            ret_candle[0] = self.start_opening
            ret_candle[1] = round(self.start_opening - candle[1], 2)  # x = start - current => current = start - x
            ret_candle[2] = round(self.start_opening - candle[2], 2)
            ret_candle[3] = round(self.start_opening - candle[3], 2)

        except ValueError:
            ret_candle[-1] = self.first_date - timedelta(minutes=int(candle[-1]))
            ret_candle[0] = round(self.start_opening - candle[0], 2)
            ret_candle[1] = round(self.start_opening - candle[1], 2)
            ret_candle[2] = round(self.start_opening - candle[2], 2)
            ret_candle[3] = round(self.start_opening - candle[3], 2)

        return ret_candle

    def get_env_state(self):
        returning_batch = []
        end_index = self.year_data_shape[0] if self.year_time_step + self.batch_size > self.year_data_shape[0] else self.year_time_step + self.batch_size
        year_data = np.memmap(self.year_data_filename, dtype=object, mode='r', shape=self.year_data_shape)
        year_indicators = np.memmap(self.indicator_class.year_indicator_filename, dtype='float32', mode='r', shape=self.indicator_class.year_indicator_shape)
        for i in range(self.year_time_step, end_index):
            individual_batch = np.array(year_data[i-60:i])
            individual_batch[:, 4] = [dt.timestamp() for dt in individual_batch[:, 4]]

            for j in year_indicators[1:]:
                individual_batch = np.column_stack((individual_batch, np.array(j[i-60:i])))
            returning_batch.append(individual_batch)
        return np.array(returning_batch, dtype=np.float32)

    # actions is a list of 256
    def step(self, actions_agents, balance_agents):
        # fetch the data for the next year
        year_data = np.memmap(self.year_data_filename, dtype=object, mode='r', shape=self.year_data_shape)
        if self.year_time_step >= self.year_data_shape[0]:
            self.current_year += 1
            if self.current_year >= self.train_end.year:
                self.done = True
            self.fetch_current_year_data()
            self.indicator_class.batch_process()

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

        returning_reward = []
        for actions_index, actions in enumerate(actions_agents):
            agent_reward = []
            # calculating the realized reward
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
                        balance_agents[actions_index] -= 1  # commission for now is 1
                        agent_reward.append(balance_agents[actions_index])
                    else:
                        if (action == 'buy' and self.orders['open'][0]['order'] == 'sell') or (action == 'sell' and self.orders['open'][0]['order'] == 'buy'):
                            move_to_close = self.orders['open'].pop()
                            move_to_close['exit_datetime'] = year_data[self.year_time_step+action_index][-1]
                            move_to_close['exit_price'] = year_data[self.year_time_step+action_index][-2]
                            self.orders['closed'].append(move_to_close)
                            # this reward below just calculates the difference between entry and exit apply a commission
                            # rate of 1 (for now)
                            reward = ((move_to_close['entry_price'] - move_to_close['exit_price']) *
                                      reward_multiplier[self.instrument]) - 1 if action == 'buy' else (
                                    ((move_to_close['exit_price'] - move_to_close['entry_price']) *
                                     reward_multiplier[self.instrument]) - 1)
                            balance_agents[actions_index] += reward
                            agent_reward.append(balance_agents[actions_index])
                        else:
                            balance_agents[actions_index] -= 1
                            agent_reward.append(balance_agents[actions_index])
                else:
                    agent_reward.append(balance_agents[actions_index])
            returning_reward.append(agent_reward)

            # self.balance = sum(returning_reward)

        # print(returning_reward)
        # updating state space to get next batch ie [:256] => [256:512]
        self.year_time_step += self.batch_size
        self.env_out = self.get_env_state()

        return [self.env_out, returning_reward]


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
