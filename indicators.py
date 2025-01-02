# somehow may also need to get in the economic canlendar in here. It also might be a good idea to just iterate through
# the entire training time and just pre pull the data rather than requesting and then working.
# Data should be the previous 60 days worth of data with the most recent being at the end of the list

import pandas as pd
import numpy as np


def calculate_ema(prices, period):
    prices = np.array(prices)

    alpha = 2 / (period + 1)

    ema = np.zeros_like(prices)
    ema[period-1] = np.mean(prices[:period])

    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema

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
        self.rsi_last_values.append(round(100 - (100 / ((avg_up/avg_down) + 1)), 2))

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
        # print(self.list_data[-1])
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


class BatchIndicators:
    def __init__(self, year_data_filename, year_data_shape, rsi_flag=False, mac_flag=False, ob_flag=False, fvg_flag=False, news_flag=False):
        self.list_data = np.memmap(year_data_filename, dtype=object, mode='r', shape=year_data_shape)
        self.rsi_flag = rsi_flag
        self.mac_flag = mac_flag
        self.ob_flag = ob_flag
        self.fvg_flag = fvg_flag
        self.news_flag = news_flag

        self.year_indicator_filename = 'year_indicator.dat'
        self.year_indicator_shape = ()

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

        self.batch_process()

    def batch_process(self):
        outputting = []
        if self.rsi_flag:
            outputting.append(self.rsi_calculate())
        if self.mac_flag:
            outputting.append(self.mac_calculate())
        # if self.ob_flag:
        #     self.ob_calculate()
        # if self.fvg_flag:
        #     self.fvg_calculate()
        # if self.news_flag:
        #     self.news_calculate()
        outputting = np.array(outputting)
        self.year_indicator_shape = outputting.shape
        arr = np.memmap(self.year_indicator_filename, dtype='float32', mode='w+', shape=self.year_indicator_shape)
        for i in range(self.year_indicator_shape[0]):
            arr[i] = outputting[i]
        arr.flush()

    def rsi_calculate(self):
        period = 14
        closes = np.array([data[3] for data in self.list_data])

        price_changes = np.diff(closes)
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)

        avg_gain = np.zeros_like(closes)
        avg_loss = np.zeros_like(closes)

        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period+1, len(closes)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i-1]) / period

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 0

        return rsi

    def mac_calculate(self):
        short_period = 12
        long_period = 26
        signal_period = 9
        closes = [data[3] for data in self.list_data]

        ema_short = calculate_ema(closes, short_period)
        ema_long = calculate_ema(closes, long_period)
        # print(ema_short, ema_long)

        # Calculate the MACD Line (difference between fast and slow EMAs)
        macd_line = np.zeros_like(closes)
        macd_line[25:] = np.array(ema_short)[25:] - np.array(ema_long)[25:]

        # Calculate the Signal Line (9-period EMA of the MACD Line)
        signal_line = np.zeros_like(closes)
        signal_line[25:] = calculate_ema(macd_line[25:], signal_period)
        # signal_line[34:] = calculate_ema(macd_line[25:], signal_period)

        # Return the MACD Line and Signal Line (same length as the input data)

        return macd_line


if __name__ == '__main__':
    test_data = [[19629,19645.25,19614.5,19623.5],
[19623.25,19649.75,19623,19641.5],
[19641,19672.75,19640.75,19667.5],
[19665.5,19673.75,19660,19661.5],
[19662.25,19665.25,19653.25,19656],
[19657,19659.25,19650.25,19654],
[19653.25,19664,19652.5,19662.25],
[19662.25,19662.75,19655.5,19657],
[19657.5,19658.75,19656,19658],
[19657,19658.75,19655,19655],
[19655,19663,19653.25,19661.75],
[19662,19663.25,19657.25,19661.25],
[19661,19663.5,19659,19660],
[19660.25,19662.75,19642.25,19645.75],
[19646.75,19654,19646.25,19653],
[19653.5,19654,19645.75,19646.5],
[19646.75,19647,19640.25,19645],
[19645.5,19647.25,19640,19640],
[19640.25,19640.5,19633,19635.5],
[19638,19648.5,19637.75,19641.5],
[19641.5,19642.75,19640,19641.25],
[19641.25,19641.25,19639.25,19639.75],
[19639.75,19642.5,19635.5,19639.75],
[19639.75,19639.75,19636.75,19637.5],
[19638.75,19641,19627.5,19631.25],
[19632.5,19633,19621,19629],
[19629.25,19636.25,19627.25,19633.75],
[19635,19647,19634.25,19637.25],
[19636.75,19642,19633,19641],
[19641,19643,19637,19638],
[19638,19643.25,19638,19641.25],
[19642.75,19647,19637.5,19637.5],
[19639,19643.25,19639,19642],
[19641.5,19643,19638.25,19643],
[19643.75,19644,19640.75,19643],
[19643.25,19644,19640,19640.5],
[19641.5,19644,19640,19644],
[19644.5,19644.5,19641.75,19643.5],
[19643.25,19644.25,19641,19644.25],
[19642.25,19645,19642.25,19645],
[19643.25,19643.25,19638,19642.5],
[19642.25,19642.25,19636.25,19640],
[19640,19640.5,19634.5,19636.5],
[19637,19638.25,19635.5,19636.5],
[19635.25,19636,19632,19633.75],
[19634.75,19638,19631,19637],
[19638,19646,19638,19644],
[19644.25,19645,19640,19641],
[19641.5,19643,19641.5,19641.5],
[19641.5,19643.5,19641.5,19643.5],
[19642.75,19644,19640,19644],
[19642.25,19648,19642.25,19647],
[19647.5,19649.25,19643.5,19646.5],
[19648.25,19651.25,19647.5,19649.5],
[19647.25,19653.75,19647.25,19653.25],
[19652,19655,19648.75,19650],
[19649.25,19652.25,19647.75,19650.5],
[19650.25,19651.75,19646.75,19650],
[19651.25,19651.5,19648.25,19648.25],
[19649,19651.5,19644.25,19646.75]]

    a = BatchIndicators(test_data, mac_flag=True)
    print(a.state_space)
    # next = [2227.9, 2227.9, 2227.9, 2227.9, '2011-01-13 01:00:00']
    # ends = []
    # a = Indicators(test_data, rsi_flag=True)
    # print(a.rsi_last_values)
    # for i in range(0, 10):
    #     start = time.time()
    #     a.rsi_init()
    #     b = time.time() - start
    #     print(b)
    #     ends.append(b)
    # print(sum(ends)/10)
    # print(a.rsi_last_values)
    # print(a.news_last_values)
    # print(a.get_indicators(next))
    # a.economic_news()
    # print(a.news_last_values)
