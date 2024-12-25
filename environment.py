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
    a = EnviroLocalTraining(instrument, train_start, train_end)
    print(a.train_data)
    print(a.env_out)
    a.step('hold')

