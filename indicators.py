# somehow may also need to get in the economic canlendar in here. It also might be a good idea to just iterate through
# the entire training time and just pre pull the data rather than requesting and then working.
# Data should be the previous 60 days worth of data with the most recent being at the end of the list

import pandas as pd
from datetime import datetime

# # This macd list will have the format of [ema_12, ema_26, macd, signal]
# def macd(data, macd_pre):
#     mac_together = []
#     macd_cur = []
#     multi_12 = 2 / 13
#     multi_26 = 2 / 27
#     multi_9 = 2 / 10
#     if len(macd_pre) == 0:
#         sma_12 = []
#         sma_26 = []
#         ema_12 = []
#         ema_26 = []
#         for i in range(len(data)):
#             if i > 12:
#                 # sma_12.append(data[i][3])
#                 avg = sum(sma_12) / len(sma_12)
#                 if i == 13:
#                     calc = data[i][3] * multi_12 + avg * (1 - multi_12)
#                 else:
#                     calc = data[i][3] * multi_12 + ema_12[-1] * (1 - multi_12)
#                 ema_12.append(calc)
#             else:
#                 sma_12.append(data[i][3])
#             if i > 26:
#                 avg = sum(sma_26) / len(sma_26)
#                 if i == 27:
#                     calc = data[i][3] * multi_26 + avg * (1 - multi_26)
#                 else:
#                     calc = data[i][3] * multi_26 + ema_26[-1] * (1 - multi_26)
#                 ema_26.append(calc)
#                 macd_cur.append(ema_12[-1] - ema_26[-1])
#                 if len(macd_cur) > 9:
#                     if len(macd_cur) == 10:
#                         avg = sum(macd_cur[:-1]) / len(macd_cur[:-1])
#                         calc = macd_cur[-1] * multi_9 + avg * (1 - multi_9)
#                     else:
#                         calc = macd_cur[-1] * multi_9 + mac_together[-1][-1] * (1 - multi_9)
#                     mac_together.append([ema_12, ema_26, macd_cur[-1], calc])
#             else:
#                 sma_26.append(data[i][3])
#     else:
#         for i in data:
#             ema_12_cur = data[i][3] * multi_12 + macd_pre[-1][0] * (1 - multi_12)
#             ema_26_cur = data[i][3] * multi_26 + macd_pre[-1][1] * (1 - multi_26)
#             macd_calc = ema_12_cur - ema_26_cur
#             signal = macd_calc * multi_9 + macd_pre[-1][2] * (1 - multi_9)
#             # macd_cur.append(data[i][3])
#             mac_together.append([ema_12_cur, ema_26_cur, macd_calc, signal])
#     return mac_together
#
#
# # ohlc
# def order_block(data):
#     obs = []
#     for i in range(len(data)):
#         if i == 0:
#             pass
#         if i == len(data) - 1:
#             break
#
#         # for lows so the order block will be a down close candle
#         if data[i][2] < data[i + 1][2] and data[i][2] < data[i - 1][2]:
#             if data[i - 1][0] > data[i - 1][3]:
#                 obs.append(data[i - 1][0])
#             elif data[i][0] < data[i][3]:
#                 obs.append(data[i][0])
#             elif data[i + 1][0] > data[i + 1][3]:
#                 obs.append(data[i + 1][0])
#         # for highs so the order block will be an up close candle
#         if data[i][3] > data[i - 1][3] and data[i][3] > data[i - 1][3]:
#             if data[i - 1][0] < data[i - 1][3]:
#                 obs.append(data[i - 1][0])
#             elif data[i][0] < data[i][3]:
#                 obs.append(data[i][0])
#             elif data[i + 1][0] < data[i + 1][3]:
#                 obs.append(data[i + 1][0])
#     return obs
#
#
# def fvg(data):
#     fvgs = []
#     for i in range(len(data)):
#         if i + 2 >= len(data):
#             break
#         if data[i][1] < data[i + 2][2]:
#             fvgs.append([data[i][1], data[i + 2][2]])
#         elif data[i][1] > data[i + 2][2]:
#             fvgs.append([data[i + 2][1], data[i][1]])
#     return fvgs

# indicators should return with a constant length array
# For the research paper method the first 60 entries consist of rsi and the next 60 consist of macd (for now forget
# about trigger line)
# For smt method consist of maybe last 5 OB+, next 5 are OB- next 5 are [fvg_low, fvg_high], 60 entries of economic
# news where 0 is nothing, 1 is low impact, 2 is medium impact, 3 is high impact and the highest always takes precedence
# ignore all day news?


class Indicators:
    def __init__(self, data, rsi_flag=False, mac_flag=False, ob_flag=False, fvg_flag=False, news_flag=False):
        self.list_data = data
        self.rsi_flag = rsi_flag
        self.mac_flag = mac_flag
        self.ob_flag = ob_flag
        self.fvg_flag = fvg_flag
        self.news_flag = news_flag

        self.rsi_need = []
        self.mac_need = []
        self.df = pd.read_csv('eco_news.csv')
        self.df['Time'] = pd.to_datetime(self.df['Time'], format='%I:%M%p')
        self.df['Time'] = self.df['Time'].dt.strftime('%H:%M:%S')

        self.rsi_last_values = []
        self.mac_last_values = [[], []]  # list of 2 items, the first is list of macd and the second is list of trigger
        self.ob_up_last_values = []
        self.ob_down_last_values = []
        self.fvgs_up_last_values = []
        self.fvgs_down_last_values = []
        self.news_last_values = []

        self.indicators_init()

    def get_state_space(self):
        outputting = []
        for i in range(60):
            appending = [self.list_data[i][-2]]
            if self.rsi_flag:
                appending.append(self.rsi_last_values[i])
            if self.mac_flag:
                appending.append(self.mac_last_values[0][i])
            # if self.ob_flag:
            #     appending.extend([self.ob_up_last_values[i], self.ob_down_last_values[i]])
            # if self.fvg_flag:
            #     appending.extend([self.fvgs_up_last_values[i], self.fvgs_down_last_values[i]])
            if self.news_flag:
                appending.append(self.news_last_values[i])
            outputting.append(appending)
        return outputting

    def indicators_init(self):
        if self.rsi_flag:
            self.rsi_init()
        if self.mac_flag:
            self.mac_init()
            # for now when calling mac will only return the mac value and not the trigger
        if self.ob_flag:
            self.ob_init()
        if self.fvg_flag:
            self.fvg_init()
        if self.news_flag:
            self.economic_news_init()

    def get_indicators(self, cur_data):
        self.list_data.pop(0)
        self.list_data.append(cur_data)

        if self.rsi_flag:
            self.rsi()
        if self.mac_flag:
            self.mac()
        if self.ob_flag:
            self.ob()
        if self.fvg_flag:
            self.fvg()
        if self.news_flag:
            self.economic_news()

    def rsi_init(self):
        working_range = 14
        upward = []
        downward = []
        for i in range(len(self.list_data)):
            if i == 0:
                continue
            if self.list_data[i][-2] > self.list_data[i - 1][-2]:
                upward.append(self.list_data[i][-2] - self.list_data[i - 1][-2])
                downward.append(0)
            else:
                downward.append(self.list_data[i - 1][-2] - self.list_data[i][-2])
                upward.append(0)
        avg_up = sum(upward[:working_range]) / len(upward[:working_range])
        avg_down = sum(downward[:working_range]) / len(downward[:working_range])

        for i in range(working_range, len(self.list_data) - 1):
            avg_up = ((avg_up * (working_range - 1) + upward[i]) / working_range)
            avg_down = ((avg_down * (working_range - 1) + downward[i]) / working_range)
            rs = avg_up / avg_down
            self.rsi_last_values.append(round(100 - (100 / (rs + 1)), 2))
        self.rsi_need = [avg_up, avg_down]
        self.rsi_last_values = [0]*(60-len(self.rsi_last_values)) + self.rsi_last_values

    def rsi(self):
        working_range = 14
        upward = 0
        downward = 0
        print(self.list_data[-1])
        if self.list_data[-1][-2] > self.list_data[-2][-2]:
            upward = self.list_data[-1][-2] - self.list_data[-2][-2]
        else:
            downward = self.list_data[-2][-2] - self.list_data[-1][-2]

        avg_up = ((self.rsi_need[0] * (working_range - 1) + upward) / working_range)
        avg_down = ((self.rsi_need[1] * (working_range - 1) + downward) / working_range)
        rs = avg_up / avg_down
        rsi_out = round(100 - (100 / (rs + 1)), 2)
        self.rsi_need = [avg_up, avg_down]

        self.rsi_last_values.pop(0)
        self.rsi_last_values.append(rsi_out)

    def mac_init(self):
        multi_12 = 2 / 13
        multi_26 = 2 / 27
        multi_9 = 2 / 10
        sma_12 = []
        sma_26 = []
        ema_12 = []
        ema_26 = []
        for i in range(len(self.list_data)):
            if i >= 12:
                if i == 12:
                    avg = sum(sma_12) / len(sma_12)
                    calc = self.list_data[i][-2] * multi_12 + avg * (1 - multi_12)
                else:
                    calc = self.list_data[i][-2] * multi_12 + ema_12[-1] * (1 - multi_12)
                ema_12.append(calc)
            else:
                sma_12.append(self.list_data[i][-2])
            if i >= 26:
                if i == 26:
                    avg = sum(sma_26) / len(sma_26)
                    self.mac_last_values[0].append(round(ema_12[-2] - avg, 2))
                    calc = self.list_data[i][-2] * multi_26 + avg * (1 - multi_26)
                else:
                    calc = self.list_data[i][-2] * multi_26 + ema_26[-1] * (1 - multi_26)
                ema_26.append(calc)
                self.mac_last_values[0].append(round(ema_12[-1] - ema_26[-1], 2))
                if len(self.mac_last_values[0]) > 9:
                    if len(self.mac_last_values[0]) == 10:
                        avg = sum(self.mac_last_values[0][:9]) / len(self.mac_last_values[0][:9])
                        calc = self.mac_last_values[0][-1] * multi_9 + avg * (1 - multi_9)
                    else:
                        calc = self.mac_last_values[0][-1] * multi_9 + self.mac_last_values[1][-1] * (1 - multi_9)
                    self.mac_last_values[1].append(round(calc, 2))
            else:
                sma_26.append(self.list_data[i][3])
        self.mac_need = [ema_12[-1], ema_26[-1]]
        self.mac_last_values[0] = [0] * (60 - len(self.mac_last_values[0])) + self.mac_last_values[0]
        self.mac_last_values[1] = [0] * (60 - len(self.mac_last_values[1])) + self.mac_last_values[1]

    def mac(self):
        multi_12 = 2 / 13
        multi_26 = 2 / 27
        multi_9 = 2 / 10

        ema_12_cur = self.list_data[-1][-2] * multi_12 + self.mac_need[0] * (1 - multi_12)
        ema_26_cur = self.list_data[-1][-2] * multi_26 + self.mac_need[1] * (1 - multi_26)
        macd_calc = ema_12_cur - ema_26_cur
        signal = macd_calc * multi_9 + self.mac_last_values[1][-1] * (1 - multi_9)

        self.mac_need = [ema_12_cur, ema_26_cur]
        self.mac_last_values[0].pop(0)
        self.mac_last_values[0].append(round(macd_calc, 2))

        self.mac_last_values[1].pop(0)
        self.mac_last_values[1].append(round(signal, 2))

    def ob_init(self):
        obs = []
        for i in range(len(self.list_data)):
            if i == 0:
                continue
            if i == len(self.list_data) - 1:
                break

            # for bullish OB first OHLC
            if self.list_data[i][2] < self.list_data[i + 1][2] and self.list_data[i][2] < self.list_data[i - 1][2]:
                # checks first for the middle if it is down close this is the one that we will want to take and is the
                # best case
                if self.list_data[i][0] > self.list_data[i][3]:
                    self.ob_up_last_values.append(self.list_data[i][0])
                    print(i, self.list_data[i][0])
                # checks if first candle is down close candle
                elif self.list_data[i - 1][0] > self.list_data[i - 1][3]:
                    self.ob_up_last_values.append(self.list_data[i - 1][0])
                    print(i, self.list_data[i - 1][0])
                # checks if 3rd candle is down close candle
                elif self.list_data[i + 1][0] > self.list_data[i + 1][3]:
                    self.ob_up_last_values.append(self.list_data[i + 1][0])
                    print(i, self.list_data[i + 1][0])
            # for bearish the order block will be an up close candle OHLC
            if self.list_data[i][1] > self.list_data[i - 1][1] and self.list_data[i][1] > self.list_data[i + 1][1]:
                # checks first for the middle it is an up close candle this is the best case and the one we will take
                if self.list_data[i][0] < self.list_data[i][3]:
                    self.ob_down_last_values.append(self.list_data[i][0])
                    print(i, self.list_data[i][0])
                elif self.list_data[i - 1][0] < self.list_data[i - 1][3]:
                    self.ob_down_last_values.append(self.list_data[i - 1][0])
                    print(i, self.list_data[i - 1][0])
                elif self.list_data[i + 1][0] < self.list_data[i + 1][3]:
                    self.ob_down_last_values.append(self.list_data[i + 1][0])
                    print(i, self.list_data[i + 1][0])
        self.ob_up_last_values = self.ob_up_last_values[-3:]
        self.ob_down_last_values = self.ob_down_last_values[-3:]

    def ob(self):
        if self.list_data[-2][2] < self.list_data[-1][2] and self.list_data[-2][2] < self.list_data[-3][2]:
            # checks first for the middle if it is down close this is the one that we will want to take and is the
            # best case
            if self.list_data[-2][0] > self.list_data[-2][3]:
                self.ob_up_last_values.pop(0)
                self.ob_up_last_values.append(self.list_data[-2][0])
            # checks if first candle is down close candle
            elif self.list_data[-3][0] > self.list_data[-3][3]:
                self.ob_up_last_values.pop(0)
                self.ob_up_last_values.append(self.list_data[-3][0])
            # checks if 3rd candle is down close candle
            elif self.list_data[-1][0] > self.list_data[-1][3]:
                self.ob_up_last_values.pop(0)
                self.ob_up_last_values.append(self.list_data[-1][0])
        # for bearish the order block will be an up close candle OHLC
        if self.list_data[-2][1] > self.list_data[-3][1] and self.list_data[-2][1] > self.list_data[-1][1]:
            # checks first for the middle it is an up close candle this is the best case and the one we will take
            if self.list_data[-2][0] < self.list_data[-2][3]:
                self.ob_down_last_values.pop(0)
                self.ob_down_last_values.append(self.list_data[-2][0])
            elif self.list_data[-3][0] < self.list_data[-3][3]:
                self.ob_down_last_values.pop(0)
                self.ob_down_last_values.append(self.list_data[-3][0])
            elif self.list_data[-1][0] < self.list_data[-1][3]:
                self.ob_down_last_values.pop(0)
                self.ob_down_last_values.append(self.list_data[-1][0])

    def fvg_init(self):
        for i in range(len(self.list_data)):
            if i + 2 >= len(self.list_data):
                break
            if self.list_data[i][1] < self.list_data[i + 2][2]:
                self.fvgs_up_last_values.append([self.list_data[i][1], self.list_data[i + 2][2]])
            elif self.list_data[i][2] > self.list_data[i + 2][1]:
                self.fvgs_down_last_values.append([self.list_data[i + 2][1], self.list_data[i][2]])

        up_num = len(self.fvgs_up_last_values)
        if up_num >= 3:
            self.fvgs_up_last_values = self.fvgs_up_last_values[-3:]
        else:
            self.fvgs_up_last_values = [0]*(3-up_num) + self.fvgs_up_last_values

        down_num = len(self.fvgs_down_last_values)
        if down_num >= 3:
            self.fvgs_down_last_values = self.fvgs_down_last_values[-3:]
        else:
            self.fvgs_down_last_values = [0]*(3-down_num) + self.fvgs_down_last_values

    def fvg(self):
        if self.list_data[-3][1] < self.list_data[-1][2]:
            self.fvgs_up_last_values.pop(0)
            self.fvgs_up_last_values.append([self.list_data[-3][1], self.list_data[-1][2]])
        elif self.list_data[-3][2] > self.list_data[-1][1]:
            self.fvgs_down_last_values.pop(0)
            self.fvgs_down_last_values.append([self.list_data[-3][1], self.list_data[-1][2]])

    def economic_news_init(self):
        for i in range(len(self.list_data)):
            print(self.list_data[i][-1])
            list_data_date = str(self.list_data[i][-1]).split(' ')[0]
            list_data_time = str(self.list_data[i][-1]).split(' ')[-1]
            current_news = self.df.loc[(self.df['Date'] == list_data_date) & (self.df['Time'] == list_data_time)]['Impact']
            if len(current_news) == 0:
                self.news_last_values.append(0)
            else:
                self.news_last_values.append(current_news.max())

    def economic_news(self):
        self.news_last_values.pop(0)

        list_data_date = str(self.list_data[-1][-1]).split(' ')[0]
        list_data_time = str(self.list_data[-1][-1]).split(' ')[-1]
        current_news = self.df.loc[(self.df['Date'] == list_data_date) & (self.df['Time'] == list_data_time)]['Impact']
        if len(current_news) == 0:
            self.news_last_values.append(0)
        else:
            self.news_last_values.append(current_news.max())


if __name__ == '__main__':
    test_data = [[2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:00:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:01:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:02:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:03:00'],
                 [2228.4, 2228.4, 2228.4, 2228.4, '2011-01-13 08:04:00'],
                 [2228.2, 2228.0, 2228.2, 2228.2, '2011-01-13 08:05:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:06:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:07:00'],
                 [2227.7, 2227.5, 2227.7, 2227.5, '2011-01-13 08:08:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:09:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:10:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:11:00'],
                 [2227.7, 2227.7, 2227.7, 2227.7, '2011-01-13 08:12:00'],
                 [2227.7, 2227.7, 2227.7, 2227.7, '2011-01-13 08:13:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:14:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:15:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:16:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:17:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:18:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:19:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:20:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:21:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:22:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:23:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:24:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:25:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:26:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:27:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:28:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:29:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:30:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:31:00'],
                 [2228.2, 2228.2, 2228.2, 2228.2, '2011-01-13 08:32:00'],
                 [2227.9, 2227.6, 2227.9, 2227.6, '2011-01-13 08:33:00'],
                 [2227.9, 2227.6, 2227.9, 2227.6, '2011-01-13 08:34:00'],
                 [2227.9, 2227.6, 2227.9, 2227.6, '2011-01-13 08:35:00'],
                 [2227.9, 2227.6, 2227.9, 2227.6, '2011-01-13 08:36:00'],
                 [2227.9, 2227.6, 2227.9, 2227.6, '2011-01-13 08:37:00'],
                 [2228.4, 2228.4, 2228.6, 2228.6, '2011-01-13 08:38:00'],
                 [2228.4, 2228.4, 2228.4, 2228.4, '2011-01-13 08:39:00'],
                 [2228.4, 2228.4, 2228.4, 2228.4, '2011-01-13 08:40:00'],
                 [2228.2, 2228.0, 2228.2, 2228.0, '2011-01-13 08:41:00'],
                 [2228.2, 2228.0, 2228.2, 2228.0, '2011-01-13 08:42:00'],
                 [2228.7, 2228.7, 2230.0, 2229.7, '2011-01-13 08:43:00'],
                 [2227.4, 2227.4, 2227.4, 2227.4, '2011-01-13 08:44:00'],
                 [2227.7, 2227.5, 2227.7, 2227.5, '2011-01-13 08:45:00'],
                 [2227.7, 2227.5, 2227.7, 2227.5, '2011-01-13 08:46:00'],
                 [2227.7, 2227.5, 2227.7, 2227.5, '2011-01-13 08:47:00'],
                 [2227.7, 2227.5, 2227.7, 2227.5, '2011-01-13 08:48:00'],
                 [2227.7, 2227.5, 2227.7, 2227.5, '2011-01-13 08:49:00'],
                 [2227.7, 2227.5, 2227.7, 2227.5, '2011-01-13 08:50:00'],
                 [2227.7, 2227.7, 2227.7, 2227.7, '2011-01-13 08:51:00'],
                 [2227.7, 2227.7, 2227.7, 2227.7, '2011-01-13 08:52:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:53:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:54:00'],
                 [2227.7, 2227.7, 2227.7, 2227.7, '2011-01-13 08:55:00'],
                 [2227.7, 2227.7, 2227.7, 2227.7, '2011-01-13 08:56:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:57:00'],
                 [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 08:58:00'],
                 [2227.7, 2227.7, 2227.7, 2227.7, '2011-01-13 08:59:00']]

    next = [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 01:00:00']
    a = Indicators(test_data, news_flag=True)
    print(a.news_last_values)
    print(a.get_indicators(next))
    a.economic_news()
    print(a.news_last_values)
