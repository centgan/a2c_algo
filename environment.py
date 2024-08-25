import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import logging
import pytz

est = pytz.timezone('America/New_York')
utc = pytz.utc

API = "api-fxpractice.oanda.com"
STREAM_API = "stream-fxpractice.oanda.com"

ACCOUNT_ID = "101-002-25776676-003"
TOKEN = "d8e783a23ff8bab21476e440b3d578ef-207d5f7676d7d82839a50d4907b3d6e6"


# Data should be the previous 60 days worth of data with the most recent being at the end of the list
def rsi(data):
    # Check if this works
    working_range = 30
    upward = []
    downward = []
    for i in range(len(data)):
        if i == 0:
            pass
        if data[i][-2] > data[i-1][-2]:
            upward.append(data[i][-2] - data[i-1][-2])
            downward.append(0)
        else:
            downward.append(data[i-1][-2] - data[i][-2])
            upward.append(0)
    avg_up = sum(upward[:working_range]) / len(upward[:working_range])
    avg_down = sum(downward[:working_range]) / len(downward[:working_range])

    rsi_out = []
    for i in range(working_range, len(data)):
        avg_up = ((avg_up * (working_range - 1) + upward[i])/working_range)
        avg_down = ((avg_down * (working_range - 1) + downward[i])/working_range)
        rs = avg_up/avg_down
        rsi_out.append(100 - (100/(rs+1)))

    return rsi_out


# This macd list will have the format of [ema_12, ema_26, macd, signal]
def macd(data, macd_pre):
    mac_together = []
    macd_cur = []
    multi_12 = 2 / 13
    multi_26 = 2 / 27
    multi_9 = 2 / 10
    if len(macd_pre) == 0:
        sma_12 = []
        sma_26 = []
        ema_12 = []
        ema_26 = []
        for i in range(len(data)):
            if i > 12:
                # sma_12.append(data[i][3])
                avg = sum(sma_12) / len(sma_12)
                if i == 13:
                    calc = data[i][3]*multi_12 + avg*(1-multi_12)
                else:
                    calc = data[i][3]*multi_12 + ema_12[-1]*(1-multi_12)
                ema_12.append(calc)
            else:
                sma_12.append(data[i][3])
            if i > 26:
                avg = sum(sma_26)/len(sma_26)
                if i == 27:
                    calc = data[i][3]*multi_26 + avg*(1-multi_26)
                else:
                    calc = data[i][3]*multi_26 + ema_26[-1]*(1-multi_26)
                ema_26.append(calc)
                macd_cur.append(ema_12[-1] - ema_26[-1])
                if len(macd_cur) > 9:
                    if len(macd_cur) == 10:
                        avg = sum(macd_cur[:-1])/len(macd_cur[:-1])
                        calc = macd_cur[-1]*multi_9 + avg*(1-multi_9)
                    else:
                        calc = macd_cur[-1]*multi_9 + mac_together[-1][-1]*(1-multi_9)
                    mac_together.append([ema_12, ema_26, macd_cur[-1], calc])
            else:
                sma_26.append(data[i][3])
    else:
        for i in data:
            ema_12_cur = data[i][3]*multi_12 + macd_pre[-1][0]*(1-multi_12)
            ema_26_cur = data[i][3]*multi_26 + macd_pre[-1][1]*(1-multi_26)
            macd_calc = ema_12_cur - ema_26_cur
            signal = macd_calc*multi_9 + macd_pre[-1][2]*(1-multi_9)
            # macd_cur.append(data[i][3])
            mac_together.append([ema_12_cur, ema_26_cur, macd_calc, signal])
    return mac_together


# ohlc
def order_block(data):
    obs = []
    for i in range(len(data)):
        if i == 0:
            pass
        if i == len(data) - 1:
            break

        # for lows so the order block will be a down close candle
        if data[i][2] < data[i+1][2] and data[i][2] < data[i-1][2]:
            if data[i-1][0] > data[i-1][3]:
                obs.append(data[i-1][0])
            elif data[i][0] < data[i][3]:
                obs.append(data[i][0])
            elif data[i+1][0] > data[i+1][3]:
                obs.append(data[i+1][0])
        # for highs so the order block will be an up close candle
        if data[i][3] > data[i-1][3] and data[i][3] > data[i-1][3]:
            if data[i-1][0] < data[i-1][3]:
                obs.append(data[i-1][0])
            elif data[i][0] < data[i][3]:
                obs.append(data[i][0])
            elif data[i+1][0] < data[i+1][3]:
                obs.append(data[i+1][0])
    return obs

def fvg(data):
    fvgs = []
    for i in range(len(data)):
        if i + 2 >= len(data):
            break
        if data[i][1] < data[i+2][2]:
            fvgs.append([data[i][1], data[i+2][2]])
        elif data[i][1] > data[i+2][2]:
            fvgs.append([data[i+2][1], data[i][1]])
    return fvgs


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
                if action == 'buy' and open_orders[0]['order'] == 'sell' or action == 'sell' and open_orders[-1]['order'] == 'buy':
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

# a = EnviroTraining('NAS100_USD', '2004-01-06', '2004-02-01')

# print(a.step('hold'))

# header = {'Authorization': 'Bearer ' + TOKEN}
# hist_path = f'/v3/accounts/{ACCOUNT_ID}/instruments/NAS100_USD/candles'
# train_start = datetime.strptime('2010-01-05 8:00:00', "%Y-%m-%d %H:%M:%S")
# train_end = train_start + timedelta(hours=6)
# from_time = time.mktime(pd.to_datetime(train_start).timetuple())
# to_time = time.mktime(pd.to_datetime(train_end).timetuple())
# # to_time = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d 00:00:00")
# # print(to_time)
# # from_time = time.mktime(pd.to_datetime(to_time).timetuple())
# # print((datetime.now() - timedelta(hours=20)).strftime("%Y-%m-%d %H:%M:%S"))
# # a = datetime.strptime(datetime.now().strftime("%Y-%m-%d 00:00:00"), "%Y-%m-%d %H:%M:%S")
# # print(a - timedelta(hours=22))
# # to_time = time.mktime(pd.to_datetime(a - timedelta(hours=18)).timetuple())
#
# query = {'from': str(from_time), 'to': str(to_time), 'granularity': 'S5'}
# response = requests.get('https://' + API + hist_path, headers=header, params=query)
# into_json = response.json()
# print(into_json)
# with open('tester.json', 'w') as write:
#     json.dump(into_json['candles'], write, indent=4)

# instrument = 'NAS100_USD'
# start = (datetime.now() - timedelta(days=15, hours=0, minutes=0)).strftime('%Y-%m-%d 00:00:00')
# end = (datetime.now() - timedelta(days=0, hours=0, minutes=0)).strftime('%Y-%m-%d 00:00:00')
# hist_path = f'/v3/accounts/{ACCOUNT_ID}/instruments/{instrument}/candles'
#
# from_time = time.mktime(pd.to_datetime(start).timetuple())
# to_time = time.mktime(pd.to_datetime(end).timetuple())
#
# header = {'Authorization': 'Bearer ' + TOKEN}
# query = {'from': str(from_time), 'to': str(to_time), 'granularity': 'D'}
#
# response = requests.get('https://' + API + hist_path, headers=header, params=query)
# into_json = response.json()
# print(into_json)
