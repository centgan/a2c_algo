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


class EnviroTraining:
    def __init__(self, instrument, train_start, train_end):
        self.instrument = instrument
        # ensure that datetime formats are in %Y-%m-% this format and assumed to be at midnight
        self.train_start = datetime.strptime(train_start, "%Y-%m-%d")
        self.train_end = self.train_start + timedelta(hours=6)
        self.current_time_step = 60
        self.train_final_end = datetime.strptime(train_end, "%Y-%m-%d")

        self.logger = logging.getLogger()
        logging.basicConfig(filename='log.log', level=logging.INFO,
                            format='%(asctime)s  %(levelname)s: %(message)s')
        self.logger.info(
            f'Start of new Training from {self.train_start} to {self.train_final_end} on {self.instrument}')

        self.train_data = self.get_data()
        # self.current_time_step = len(self.train_data)
        # for i in range(len(self.train_data)):
        #     if i + 1 != len(self.train_data):
        #         cur_datetime = datetime.strptime(self.train_data[i][-1][:19], '%Y-%m-%d %H:%M:%S')
        #         future_datetime = datetime.strptime(self.train_data[i+1][-1][:19], '%Y-%m-%d %H:%M:%S')
        #         n = (future_datetime - cur_datetime)
        #         if n != timedelta(seconds=5):
        #             print('ERROR at', cur_datetime)
        # with open('tester.json', 'w') as f:
        #     json.dump(self.train_data, f, indent=4)
        self.logger.info(f'Data pull from {self.train_start} to {self.train_end}')
        # where orders are logged in the format {'open': [{datetime, entry_price, order}], 'closed': [{datetime,
        # entry_price, close_price, order}]}
        self.orders = {'open': [], 'closed': []}
        with open('orders.json', 'w') as f:
            f.truncate()
        self.done = False

        # for x in self.train_data[self.current_time_step - 59:self.current_time_step+1]:
        #     print(x)

        # Observation space is here and can try combinations like retail: MACD + RSI, or smc: Order Blocks + FVG both
        # with time and the past 60 candles of information. There are other combos that retail use that I can try
        # out as well
        self.env_out = [[x[-2] for x in self.train_data[self.current_time_step - 60:self.current_time_step]]]

    def get_data(self):
        header = {'Authorization': 'Bearer ' + TOKEN}
        hist_path = f'/v3/accounts/{ACCOUNT_ID}/instruments/' + self.instrument + '/candles'

        from_time = time.mktime(pd.to_datetime(self.train_start).timetuple())
        to_time = time.mktime(pd.to_datetime(self.train_end).timetuple())
        # most I can work with is on the 5-second chart and can only work in 6 hour intervals.
        query = {'from': str(from_time), 'to': str(to_time), 'granularity': 'S5'}

        response = requests.get('https://' + API + hist_path, headers=header, params=query)
        into_json = response.json()
        # with open('tester_gaps.json', 'w') as f:
        #     json.dump(into_json['candles'], f, indent=4)
        # print(into_json)

        # this is acting as a temporary fix
        if self.train_end.weekday() == 5 and self.train_end.hour == 0 and self.train_start.weekday() == 4 and self.train_start.hour == 18:
            self.train_start = self.train_start + timedelta(hours=48)
            self.train_end = self.train_start + timedelta(hours=6)

            from_time = time.mktime(pd.to_datetime(self.train_start).timetuple())
            to_time = time.mktime(pd.to_datetime(self.train_end).timetuple())
            # most I can work with is on the 5-second chart and can only work in 6 hour intervals.
            query = {'from': str(from_time), 'to': str(to_time), 'granularity': 'S5'}

            response = requests.get('https://' + API + hist_path, headers=header, params=query)
            into_json = response.json()
        replace = into_json['candles'][0].copy()
        replace['time'] = str(datetime.strftime(self.train_start + timedelta(hours=5), '%Y-%m-%dT%H:%M:%S'))
        into_json['candles'].insert(0, replace)

        replace = into_json['candles'][-1].copy()
        replace['time'] = str(datetime.strftime(self.train_end + timedelta(hours=5), '%Y-%m-%dT%H:%M:%S'))
        into_json['candles'].append(replace)
        # print(into_json)

        candles = []
        for i, candle in enumerate(into_json['candles']):
            cur_datetime = datetime.strptime(candle['time'][:19], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=utc).astimezone(
                est)
            if i != len(into_json['candles']) - 1:
                future_datetime = datetime.strptime(into_json['candles'][i + 1]['time'][:19],
                                                    '%Y-%m-%dT%H:%M:%S').replace(tzinfo=utc).astimezone(est)
                n = (future_datetime - cur_datetime) / timedelta(seconds=5) - 1
                repeated = [float(candle['mid']['o']), float(candle['mid']['h']), float(candle['mid']['l']),
                            float(candle['mid']['c']), str(cur_datetime)]
                if int(n) != n:
                    self.logger.warning('Delta time is not a factor of 5')
                # if int(n) != 0:
                candles.append(repeated.copy())

                for j in range(int(n)):
                    repeated[-1] = str(cur_datetime + timedelta(seconds=5 * (j + 1)))
                    candles.append(repeated.copy())

        # with open('tester_no_gaps.json', 'w') as write:
        # json.dump(candles, write, indent=4)
        return candles

    # action must be type buy, sell, hold where hold can also mean do nothing
    def step(self, action):
        # This means that I have to be calling get_data 4 times per day in training.
        if self.current_time_step >= len(self.train_data):
            print(self.current_time_step)
            self.train_start = self.train_end
            self.train_end = self.train_start + timedelta(hours=6)
            if self.train_end >= self.train_final_end:
                self.logger.info('Training cycle has completed...')
                # print('Training cycle has completed...')
                self.done = True
                return 'break'
            # print(self.get_data())
            to_add = self.train_data[-60:].copy()
            # print(to_add)
            self.train_data = to_add + self.get_data()
            # print(self.train_data)
            self.current_time_step = 60
            self.logger.info(f'Data pull from {self.train_start} to {self.train_end}')

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
                    'datetime': self.train_data[self.current_time_step][-1],
                    'entry_price': self.train_data[self.current_time_step][-2],
                    'order': action
                })

            else:
                if ((action == 'buy' and open_orders[0]['order'] == 'sell') or
                        (action == 'sell' and open_orders[-1]['order'] == 'buy')):
                    to_append = open_orders.pop()
                    to_append['close_price'] = self.train_data[self.current_time_step][-2]
                    self.orders['closed'].append(to_append)
                    if action == 'buy':
                        reward_real = ((to_append['entry_price'] - self.train_data[self.current_time_step][-2]) *
                                       reward_multiplier[self.instrument]) - 1
                    else:
                        reward_real = ((self.train_data[self.current_time_step][-2] - to_append['entry_price']) *
                                       reward_multiplier[self.instrument]) - 1
                else:
                    # throw error
                    # self.logger.error('A trade is already open can not open another one')
                    pass

        # return the state space
        # In the state space want 60 previous closes, technical indicators (MACD, RSI) and potential expand into areas
        # such as FVG or OB.
        # self.train_data[self.current_time_step-60: self.current_time_step]

        # will deal with technical indicators at a later time
        # Observation space is here and can try combinations like retail: MACD + RSI, or smc: Order Blocks + FVG both
        # with time and the past 60 candles of information. There are other combos that retail use that I can try
        # out as well
        env_outputs = [[x[-2] for x in self.train_data[self.current_time_step - 60:self.current_time_step]]

                       ]

        # calculate the unrealized rewards which is the reward being fed into the NN
        # E-mini NASDAQ is $20/handle, micro E-mini NASDAQ is $2/handle, E-mini S&P is $50/handle, micro E-mini S&P is
        # $5/handle
        # Will only start with the micro contract sizes
        # I also add in a -1 as a commission trade in hopes that it doesn't over-trade
        reward_unreal = 0

        if len(open_orders) > 0:
            if open_orders[0]['order'] == 'buy':
                reward_unreal = ((self.train_data[self.current_time_step][-2] - open_orders[0]['entry_price']) *
                                 reward_multiplier[self.instrument]) - 1
            elif open_orders[0]['order'] == 'sell':
                reward_unreal = ((open_orders[0]['entry_price'] - self.train_data[self.current_time_step][-2]) *
                                 reward_multiplier[self.instrument]) - 1

        # print(self.train_data[self.current_time_step][-1][:19])
        # print(self.train_data[self.current_time_step+1][-1][:19])
        # ensuring that there is a consistent time step of 5S intervals since data sometimes has 10-30second jumps
        # a = datetime.strptime(self.train_data[self.current_time_step][-1][:19], '%Y-%m-%dT%H:%M:%S')
        # b = datetime.strptime(self.train_data[self.current_time_step + 1][-1][:19], '%Y-%m-%dT%H:%M:%S')
        # if (b - a) > timedelta(seconds=5):
        #     self.train_data[self.current_time_step][-1] = str((a + timedelta(seconds=5)).strftime('%Y-%m-%dT%H:%M:%S'))
        # else:
        #     self.current_time_step += 1
        # print(self.current_time_step)
        # print([x[-1] for x in self.train_data[self.current_time_step - 59:self.current_time_step + 1]])
        # print(self.orders)
        with open('orders.json', 'w') as write:
            json.dump(self.orders, write, indent=4)
        self.current_time_step += 1
        return [env_outputs, reward_real, reward_unreal]


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
        with open(f'full_training_data_{self.current_year}.json', 'r') as f:
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
            within_counter = 0
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
    # train_start = datetime.strptime(train_start, "%Y-%m-%d %H:%M:%S")
    # train_final_end = datetime.strptime(train_end, "%Y-%m-%d %H:%M:%S")
    # train_end = train_start
    instrument = 'NAS100_USD'
    a = EnviroLocalTraining(instrument, train_start, train_end)
    a.step('hold')
