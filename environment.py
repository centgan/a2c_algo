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
        self.year_time_step = 0  # keeps track of which index in the year currently on
        self.last_reward = 0
        self.done = False

        self.year_data = self.fetch_current_year_data()
        indicator_class = indicators.BatchIndicators(self.year_data, rsi_flag=True, mac_flag=True)
        self.year_indicators = indicator_class.state_space

        self.env_out = self.get_env_state()

    def fetch_current_year_data(self, year=None):
        self.current_year = year if year else self.current_year
        with open(f'./data/full_training_data_{self.current_year}.json', 'r') as f:
            # data is of json type because iterating through it all is not that slow.
            data = json.load(f)

        self.first_date = datetime.strptime(data[0][-1][:19], '%Y-%m-%d %H:%M:%S')
        uncompressed_data = []
        # padded = np.array([sublist + [None] * (5 - len(sublist)) for sublist in data], dtype=object)
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

        return uncompressed_data

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
        end_index = len(self.year_data) if self.year_time_step + self.batch_size > len(self.year_data) else self.year_time_step + self.batch_size
        for i in range(self.year_time_step, end_index):
            individual_batch = np.array(self.year_data[i:i+60])
            individual_batch[:, 4] = [dt.timestamp() for dt in individual_batch[:, 4]]

            individual_batch = np.column_stack((individual_batch, np.array(self.year_indicators[0][0][i:i+60]))) \
                if len(self.year_indicators[0]) == 2 \
                else np.column_stack((individual_batch, np.array(self.year_indicators[0][i:i+60])))

            for j in self.year_indicators[1:]:
                if len(j) == 2:
                    # this is [macd, signal] only worrying about macd right now
                    individual_batch = np.column_stack((individual_batch, np.array(j[0][i:i+60])))
                else:
                    individual_batch = np.column_stack((individual_batch, np.array(j[i:i+60])))
            returning_batch.append(individual_batch)
        return np.array(returning_batch, dtype=np.float32)

    # actions is a list of 256
    def step(self, actions):
        # fetch the data for the next year
        if self.year_time_step >= len(self.year_data):
            self.current_year += 1
            if self.current_year >= self.train_end.year:
                self.done = True
            self.year_data = self.fetch_current_year_data()

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
        # calculating the realized reward
        for action_index, action in enumerate(actions):
            if action != 'hold':
                if len(self.orders['open']) == 0:
                    self.orders['open'].append({
                        'entry_datetime': self.year_data[self.year_time_step+action_index][-1],
                        'exit_datetime': '',
                        'entry_price': self.year_data[self.year_time_step+action_index][-2],
                        'exit_price': '',
                        'order': action
                    })
                    returning_reward.append(-1)  # commission for now is 1
                else:
                    if (action == 'buy' and self.orders['open'][0]['order'] == 'sell') or (action == 'sell' and self.orders['open'][0]['order'] == 'buy'):
                        move_to_close = self.orders['open'].pop()
                        move_to_close['exit_datetime'] = self.year_data[self.year_time_step+action_index][-1]
                        move_to_close['exit_price'] = self.year_data[self.year_time_step+action_index][-2]
                        self.orders['closed'].append(move_to_close)
                        # this reward below just calculates the difference between entry and exit apply a commission
                        # rate of 1 (for now) and append on the last self.last_reward which in a sense is just the
                        # overall balance
                        reward = ((move_to_close['entry_price'] - move_to_close['exit_price']) *
                                  reward_multiplier[self.instrument]) - 1 + self.last_reward if action == 'buy' else (
                                ((move_to_close['exit_price'] - move_to_close['entry_price']) *
                                 reward_multiplier[self.instrument]) - 1 + self.last_reward)
                        returning_reward.append(reward)
                    else:
                        returning_reward.append(-1)
            else:
                returning_reward.append(0)
            self.last_reward = sum(returning_reward)

        # updating state space to get next batch ie [:256] => [256:512]
        self.year_time_step += self.batch_size
        self.env_out = self.get_env_state()

        return [self.env_out, returning_reward]


class EnviroLocalTraining:
    def __init__(self, instrument, train_start, train_end):
        self.instrument = instrument
        # ensure that datetime formats are in %Y-%m-% this format and assumed to be at midnight
        self.train_start = datetime.strptime(train_start, "%Y-%m-%d")
        # self.train_end = self.train_start + timedelta(hours=6)
        self.current_time_step = 0
        self.year_time_step = 0
        self.current_year = self.train_start.year
        self.train_end = datetime.strptime(train_end, "%Y-%m-%d")

        self.start_opening = 0
        self.first_start = ""
        self.start_date = ""
        self.year_data = self.get_data()
        self.overflow = []
        self.train_data = self.init_just_in_time()

        self.logger = logging.getLogger()
        logging.basicConfig(filename='log.log', level=logging.INFO,
                            format='%(asctime)s  %(levelname)s: %(message)s')
        self.logger.info(
            f'Start of new Training from {self.train_start} to {self.train_end} on {self.instrument}')

        self.logger.info(f'Data pull from {self.train_start} to {self.train_end}')

        # where orders are logged in the format {'open': [{datetime, entry_price, order}], 'closed': [{datetime,
        # entry_price, close_price, order}]}
        self.orders = {'open': [], 'closed': []}
        with open('orders.json', 'w') as f:
            f.truncate()
        self.done = False

        # Observation space is here and can try combinations like retail: MACD + RSI, or smc: Order Blocks + FVG both
        # with time and the past 60 candles of information. There are other combos that retail use that I can try
        # out as well
        self.state_space = indicators.Indicators(self.train_data, rsi_flag=True, mac_flag=True)
        self.env_out = self.state_space.get_state_space()

    def get_data(self):
        with open(f'./data/full_training_data_{self.current_year}.json', 'r') as f:
            return json.load(f)

    def init_just_in_time(self):
        overall = []
        while len(overall) <= 60:
            overall.extend(self.uncompress(self.year_data[self.year_time_step]))
            self.current_time_step = len(overall)
            self.year_time_step += 1
        returning_list = overall[:60]
        self.overflow = overall[60:]
        return returning_list

    def uncompress(self, data):
        ret_list = data.copy()
        if len(data) == 5:
            self.start_opening = data[0]
            self.start_date = datetime.strptime(data[-1][:19], '%Y-%m-%d %H:%M:%S')
            if self.current_time_step == 0:
                self.first_start = datetime.strptime(data[-1][:19], '%Y-%m-%d %H:%M:%S')
            ret_list[0] = self.start_opening
        else:
            ret_list[0] = round(self.start_opening - ret_list[0], 1)
        ret_list[1] = round(ret_list[0] + ret_list[1], 1)
        ret_list[2] = round(ret_list[0] + ret_list[2], 1)

        index_multi = str(data[3]).find('_x')
        if index_multi != -1:
            multiplier = int(str(data[3]).split('_x')[-1])
            ret_list[3] = round(ret_list[0] + float(str(ret_list[3]).split('_x')[0]), 1)

            full = []
            for index in range(multiplier):
                appending = ret_list.copy()
                if len(ret_list) == 5:
                    if index != 0:
                        appending[-1] = str(self.first_start + timedelta(minutes=(index + self.current_time_step)))
                    else:
                        appending[-1] = str(self.start_date)
                else:
                    appending.append(str(self.first_start + timedelta(minutes=(index + self.current_time_step))))
                full.append(appending)
            return full
        else:
            ret_list[3] = round(ret_list[0] + ret_list[3], 1)
            if len(data) == 5:
                ret_list[-1] = str(self.start_date)
            else:
                ret_list.append(str(self.first_start + timedelta(minutes=self.current_time_step)))

        return [ret_list]

    def step(self, action):
        if self.year_time_step >= len(self.year_data):
            self.current_year += 1
            self.year_data = self.get_data()
        if len(self.overflow) == 0:
            self.overflow.extend(self.uncompress(self.year_data[self.year_time_step]))
            self.year_time_step += 1
            self.current_time_step += len(self.overflow)

        open_orders = self.orders['open']

        # calculate the realized rewards which is the reward used as a metric for me
        # E-mini NASDAQ is $20/handle, micro E-mini NASDAQ is $2/handle, E-mini S&P is $50/handle, micro E-mini S&P is
        # $5/handle
        # Will only start with the micro contract sizes
        reward_multiplier = {
            # 'Micro E-mini NASDAQ': 2,
            # 'E-mini NASDAQ': 20,
            # 'Micro E-mini S&P': 5,
            # 'E-mini S&P': 50,
            'SPX500_USD': 5,
            'NAS100_USD': 2
        }

        # log and perform action
        # Subtracting 1 also from here due to commission
        reward_real = 0
        if action != 'hold':
            if len(open_orders) == 0:
                self.orders['open'].append({
                    'datetime': self.train_data[-1][-1],
                    'entry_price': self.train_data[-1][-2],
                    'order': action
                })

            else:
                if ((action == 'buy' and open_orders[0]['order'] == 'sell') or
                        (action == 'sell' and open_orders[-1]['order'] == 'buy')):
                    to_append = open_orders.pop()
                    to_append['close_price'] = self.train_data[-1][-2]
                    self.orders['closed'].append(to_append)
                    if action == 'buy':
                        reward_real = ((to_append['entry_price'] - self.train_data[-1][-2]) *
                                       reward_multiplier[self.instrument]) - 1
                    else:
                        reward_real = ((self.train_data[-1][-2] - to_append['entry_price']) *
                                       reward_multiplier[self.instrument]) - 1
                else:
                    pass

        self.train_data.pop(0)
        current_data = self.overflow.pop(0)
        self.train_data.append(current_data)
        self.state_space.get_indicators(current_data)
        env_out = self.state_space.get_state_space()
        reward_unreal = 0

        if len(open_orders) > 0:
            if open_orders[0]['order'] == 'buy':
                reward_unreal = ((self.train_data[-1][-2] - open_orders[0]['entry_price']) *
                                 reward_multiplier[self.instrument]) - 1
            elif open_orders[0]['order'] == 'sell':
                reward_unreal = ((open_orders[0]['entry_price'] - self.train_data[-1][-2]) *
                                 reward_multiplier[self.instrument]) - 1

        with open('orders.json', 'w') as write:
            json.dump(self.orders, write, indent=4)
        return [env_out, reward_real, reward_unreal]


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

