# somehow may also need to get in the economic canlendar in here. It also might be a good idea to just iterate through
# the entire training time and just pre pull the data rather than requesting and then working.
# Data should be the previous 60 days worth of data with the most recent being at the end of the list

import os
import pandas as pd
import numpy as np
import datetime


def calculate_ema(prices, period):
    prices = np.array(prices)

    alpha = 2 / (period + 1)

    ema = np.zeros_like(prices)
    ema[period - 1] = np.mean(prices[:period])

    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


# indicators should return with a constant length array
# For the research paper method the first 60 entries consist of rsi and the next 60 consist of macd (for now forget
# about trigger line)
# For smt method consist of maybe last 5 OB+, next 5 are OB- next 5 are [fvg_low, fvg_high], 60 entries of economic
# news where 0 is nothing, 1 is low impact, 2 is medium impact, 3 is high impact and the highest always takes precedence
# ignore all day news?

class BatchJITIndicator:
    def __init__(self, chunk_data, indicator_flags, testing=False):
        self.chunk_data = chunk_data  # should be a numpy array
        self.rsi_flag = indicator_flags[0]
        self.mac_flag = indicator_flags[1]
        self.ob_flag = indicator_flags[2]
        self.fvg_flag = indicator_flags[3]
        self.news_flag = indicator_flags[4]
        self.testing = testing

        self.indicator_out = {}

        self.eco_news = pd.read_csv('eco_news.csv')
        self.eco_news['Time'] = pd.to_datetime(self.eco_news['Time'], format='%I:%M%p')
        self.eco_news['Time'] = self.eco_news['Time'].dt.strftime('%H:%M:%S')
        self.eco_news['Datetime'] = pd.to_datetime(self.eco_news['Date'] + ' ' + self.eco_news['Time'])

        self.process(self.chunk_data)

    def process(self, chunk_data):
        self.chunk_data = chunk_data
        if self.rsi_flag:
            period = 14
            closes = np.array([data[3] for data in self.chunk_data])
            price_changes = np.diff(closes)
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)

            avg_gain = np.zeros_like(closes)
            avg_loss = np.zeros_like(closes)

            avg_gain[period] = np.mean(gains[:period])
            avg_loss[period] = np.mean(losses[:period])

            for i in range(period + 1, len(closes)):
                avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
                avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

            rs = np.zeros_like(closes)
            rsi = np.zeros_like(closes)

            nonzero_loss_indices = avg_loss != 0
            rs[nonzero_loss_indices] = avg_gain[nonzero_loss_indices] / avg_loss[nonzero_loss_indices]

            rsi[nonzero_loss_indices] = 100 - (100 / (1 + rs[nonzero_loss_indices]))
            rsi[~nonzero_loss_indices] = 0
            if 'rsi' in self.indicator_out.keys():
                self.indicator_out['rsi'] = np.append(self.indicator_out['rsi'][-60:], rsi[60:])
            else:
                self.indicator_out['rsi'] = rsi

        if self.mac_flag:
            short_period = 12
            long_period = 26
            signal_period = 9
            closes = [data[3] for data in self.chunk_data]

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

            if 'macd' in self.indicator_out.keys():
                self.indicator_out['macd'] = np.append(self.indicator_out['macd'][-60:], macd_line[60:])
            else:
                self.indicator_out['macd'] = macd_line

        if self.ob_flag:
            overall_ob = []
            ob_up_values = [0, 0, 0, 0, 0]  # end result shape should be (length of data, 10)
            ob_down_values = [0, 0, 0, 0, 0]
            overall_ob.append(ob_up_values + ob_down_values)
            for i in range(1, len(self.chunk_data) - 1):
                if self.chunk_data[i][2] < self.chunk_data[i + 1][2] and self.chunk_data[i][2] < self.chunk_data[i - 1][2]:
                    # checks first for the middle if it is down close this is the one that we will want to take and is the
                    # best case
                    if self.chunk_data[i][0] > self.chunk_data[i][3]:
                        ob_up_values.pop(0)
                        ob_up_values.append(self.chunk_data[i][0])
                    # checks if first candle is down close candle
                    elif self.chunk_data[i - 1][0] > self.chunk_data[i - 1][3]:
                        ob_up_values.pop(0)
                        ob_up_values.append(self.chunk_data[i - 1][0])
                    # checks if 3rd candle is down close candle
                    elif self.chunk_data[i + 1][0] > self.chunk_data[i + 1][3]:
                        ob_up_values.pop(0)
                        ob_up_values.append(self.chunk_data[i + 1][0])
                # for bearish the order block will be an up close candle OHLC
                if self.chunk_data[i][1] > self.chunk_data[i - 1][1] and self.chunk_data[i][1] > self.chunk_data[i + 1][1]:
                    # checks first for the middle it is an up close candle this is the best case and the one we will take
                    if self.chunk_data[i][0] < self.chunk_data[i][3]:
                        ob_down_values.pop(0)
                        ob_down_values.append(self.chunk_data[i][0])
                    elif self.chunk_data[i - 1][0] < self.chunk_data[i - 1][3]:
                        ob_down_values.pop(0)
                        ob_down_values.append(self.chunk_data[i - 1][0])
                    elif self.chunk_data[i + 1][0] < self.chunk_data[i + 1][3]:
                        ob_down_values.pop(0)
                        ob_down_values.append(self.chunk_data[i + 1][0])
                overall_ob.append(ob_up_values + ob_down_values)
            overall_ob.append(ob_up_values + ob_down_values)

            if 'ob' in self.indicator_out.keys():
                print(self.indicator_out['ob'])
                self.indicator_out['ob'] = self.indicator_out['ob'][-60:] + overall_ob[-60:]
            else:
                print(overall_ob[-60:])
                self.indicator_out['ob'] = overall_ob[-60:]

        if self.fvg_flag:
            overall_fvgs = []
            fvgs_up_values_hold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            fvgs_down_values_hold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # this ends up being of shape (len, 20)
            for i in range(2, len(self.chunk_data)):
                high = self.chunk_data[i - 2, 1]  # High of current
                low = self.chunk_data[i - 2, 2]  # Low of current
                high_next = self.chunk_data[i, 1]  # High of future
                low_next = self.chunk_data[i, 2]  # Low of future
                if low < high_next:
                    fvgs_up_values_hold = fvgs_up_values_hold[2:]
                    fvgs_up_values_hold.extend([low, high_next])
                elif high > low_next:
                    fvgs_down_values_hold = fvgs_down_values_hold[2:]
                    fvgs_down_values_hold.extend([high, low_next])

                overall_fvgs.append(fvgs_up_values_hold + fvgs_down_values_hold)
            # append it 2 more times so that the last 2 candles also have access to the fvgs
            overall_fvgs.append(fvgs_up_values_hold + fvgs_down_values_hold)
            overall_fvgs.append(fvgs_up_values_hold + fvgs_down_values_hold)

            if 'fvg' in self.indicator_out.keys():
                self.indicator_out['fvg'] = self.indicator_out['fvg'][-60:] + overall_fvgs[-60:]
            else:
                self.indicator_out['fvg'] = overall_fvgs[-60:]

        if self.news_flag:
            news_values = []
            eco_news_max_impact = self.eco_news.groupby('Datetime')['Impact'].max()
            eco_news_dict = eco_news_max_impact.to_dict()
            for i in self.chunk_data:
                from_timestamp = datetime.datetime.fromtimestamp(float(i[-1]))
                current_news = eco_news_dict.get(from_timestamp, None)
                if current_news is None:
                    news_values.append(0)
                else:
                    news_values.append(current_news)

            if 'news' in self.indicator_out.keys():
                self.indicator_out['news'] = self.indicator_out['news'][-60:] + news_values[60:]
            else:
                self.indicator_out['news'] = news_values[60:]



class BatchIndicators:
    def __init__(self, year_data_filename, year_data_shape, indicator_flag, testing=False):

        self.year_data_shape = year_data_shape
        self.list_data = np.memmap(year_data_filename, dtype=np.float64, mode='r', shape=self.year_data_shape)
        self.rsi_flag = indicator_flag[0]
        self.mac_flag = indicator_flag[1]
        self.ob_flag = indicator_flag[2]
        self.fvg_flag = indicator_flag[3]
        self.news_flag = indicator_flag[4]

        self.indicator_directory = './indicators/test/' if testing else './indicators/train/'
        if not os.path.exists(self.indicator_directory):
            os.makedirs(self.indicator_directory, exist_ok=True)
        self.year_indicator_length = 2580736 if testing else 4777984

        self.rsi_need = []
        self.mac_need = []
        self.eco_news = pd.read_csv('eco_news.csv')
        self.eco_news['Time'] = pd.to_datetime(self.eco_news['Time'], format='%I:%M%p')
        self.eco_news['Time'] = self.eco_news['Time'].dt.strftime('%H:%M:%S')
        self.eco_news['Datetime'] = pd.to_datetime(self.eco_news['Date'] + ' ' + self.eco_news['Time'])

        self.batch_process()

    def batch_process(self):
        if self.rsi_flag:
            if not os.path.isfile(self.indicator_directory + 'rsi_data.dat'):
                arr = np.memmap(self.indicator_directory + 'rsi_data.dat', dtype=np.float64, mode='w+',
                                shape=(self.year_indicator_length,))
                rsi_data = self.rsi_calculate()
                arr[:] = rsi_data
                arr.flush()
        if self.mac_flag:
            if not os.path.isfile(self.indicator_directory + 'mac_data.dat'):
                arr = np.memmap(self.indicator_directory + 'mac_data.dat', dtype=np.float64, mode='w+',
                                shape=(self.year_indicator_length,))
                mac_data = self.mac_calculate()
                arr[:] = mac_data
                arr.flush()
        if self.ob_flag:
            if not os.path.isfile(self.indicator_directory + 'ob_data.dat'):
                arr = np.memmap(self.indicator_directory + 'ob_data.dat', dtype=np.float64, mode='w+',
                                shape=(self.year_indicator_length, 10))
                ob_data = self.ob_calculate()
                for i in range(len(ob_data)):
                    arr[i] = ob_data[i]
                arr.flush()
        if self.fvg_flag:
            if not os.path.isfile(self.indicator_directory + 'fvg_data.dat'):
                arr = np.memmap(self.indicator_directory + 'fvg_data.dat', dtype=np.float64, mode='w+',
                                shape=(self.year_indicator_length, 20))
                fvg_data = self.fvg_calculate()
                for i in range(len(fvg_data)):
                    arr[i] = fvg_data[i]
                arr.flush()
        if self.news_flag:
            if not os.path.isfile(self.indicator_directory + 'news_data.dat'):
                arr = np.memmap(self.indicator_directory + 'news_data.dat', dtype=np.float64, mode='w+',
                                shape=(self.year_indicator_length,))
                news_data = self.news_calculate()
                arr[:] = news_data
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

        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

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

    def ob_calculate(self):
        overall_ob = []
        ob_up_values = [0, 0, 0, 0, 0]  # end result shape should be (length of data, 10)
        ob_down_values = [0, 0, 0, 0, 0]
        overall_ob.append(ob_up_values + ob_down_values)
        for i in range(1, self.year_data_shape[0] - 1):
            if self.list_data[i][2] < self.list_data[i + 1][2] and self.list_data[i][2] < self.list_data[i - 1][2]:
                # checks first for the middle if it is down close this is the one that we will want to take and is the
                # best case
                if self.list_data[i][0] > self.list_data[i][3]:
                    ob_up_values.pop(0)
                    ob_up_values.append(self.list_data[i][0])
                # checks if first candle is down close candle
                elif self.list_data[i - 1][0] > self.list_data[i - 1][3]:
                    ob_up_values.pop(0)
                    ob_up_values.append(self.list_data[i-1][0])
                # checks if 3rd candle is down close candle
                elif self.list_data[i + 1][0] > self.list_data[i + 1][3]:
                    ob_up_values.pop(0)
                    ob_up_values.append(self.list_data[i+1][0])
            # for bearish the order block will be an up close candle OHLC
            if self.list_data[i][1] > self.list_data[i - 1][1] and self.list_data[i][1] > self.list_data[i + 1][1]:
                # checks first for the middle it is an up close candle this is the best case and the one we will take
                if self.list_data[i][0] < self.list_data[i][3]:
                    ob_down_values.pop(0)
                    ob_down_values.append(self.list_data[i][0])
                elif self.list_data[i - 1][0] < self.list_data[i - 1][3]:
                    ob_down_values.pop(0)
                    ob_down_values.append(self.list_data[i-1][0])
                elif self.list_data[i + 1][0] < self.list_data[i + 1][3]:
                    ob_down_values.pop(0)
                    ob_down_values.append(self.list_data[i+1][0])
            overall_ob.append(ob_up_values + ob_down_values)
        overall_ob.append(ob_up_values + ob_down_values)

        return overall_ob

    def fvg_calculate(self):
        overall_fvgs = []
        fvgs_up_values_hold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        fvgs_down_values_hold = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # this ends up being of shape (len, 20)
        for i in range(2, self.year_data_shape[0]):
            high = self.list_data[i - 2, 1]  # High of current
            low = self.list_data[i - 2, 2]  # Low of current
            high_next = self.list_data[i, 1]  # High of future
            low_next = self.list_data[i, 2]  # Low of future
            if low < high_next:
                fvgs_up_values_hold = fvgs_up_values_hold[2:]
                fvgs_up_values_hold.extend([low, high_next])
            elif high > low_next:
                fvgs_down_values_hold = fvgs_down_values_hold[2:]
                fvgs_down_values_hold.extend([high, low_next])

            overall_fvgs.append(fvgs_up_values_hold + fvgs_down_values_hold)
        # append it 2 more times so that the last 2 candles also have access to the fvgs
        overall_fvgs.append(fvgs_up_values_hold + fvgs_down_values_hold)
        overall_fvgs.append(fvgs_up_values_hold + fvgs_down_values_hold)

        return overall_fvgs

    def news_calculate(self):
        news_values = []
        eco_news_max_impact = self.eco_news.groupby('Datetime')['Impact'].max()
        eco_news_dict = eco_news_max_impact.to_dict()
        for i in self.list_data:
            from_timestamp = datetime.datetime.fromtimestamp(float(i[-1]))
            current_news = eco_news_dict.get(from_timestamp, None)
            if current_news is None:
                news_values.append(0)
            else:
                news_values.append(current_news)

        return news_values


if __name__ == '__main__':
    test_data = [[19629, 19645.25, 19614.5, 19623.5],
                 [19623.25, 19649.75, 19623, 19641.5],
                 [19641, 19672.75, 19640.75, 19667.5],
                 [19665.5, 19673.75, 19660, 19661.5],
                 [19662.25, 19665.25, 19653.25, 19656],
                 [19657, 19659.25, 19650.25, 19654],
                 [19653.25, 19664, 19652.5, 19662.25],
                 [19662.25, 19662.75, 19655.5, 19657],
                 [19657.5, 19658.75, 19656, 19658],
                 [19657, 19658.75, 19655, 19655],
                 [19655, 19663, 19653.25, 19661.75],
                 [19662, 19663.25, 19657.25, 19661.25],
                 [19661, 19663.5, 19659, 19660],
                 [19660.25, 19662.75, 19642.25, 19645.75],
                 [19646.75, 19654, 19646.25, 19653],
                 [19653.5, 19654, 19645.75, 19646.5],
                 [19646.75, 19647, 19640.25, 19645],
                 [19645.5, 19647.25, 19640, 19640],
                 [19640.25, 19640.5, 19633, 19635.5],
                 [19638, 19648.5, 19637.75, 19641.5],
                 [19641.5, 19642.75, 19640, 19641.25],
                 [19641.25, 19641.25, 19639.25, 19639.75],
                 [19639.75, 19642.5, 19635.5, 19639.75],
                 [19639.75, 19639.75, 19636.75, 19637.5],
                 [19638.75, 19641, 19627.5, 19631.25],
                 [19632.5, 19633, 19621, 19629],
                 [19629.25, 19636.25, 19627.25, 19633.75],
                 [19635, 19647, 19634.25, 19637.25],
                 [19636.75, 19642, 19633, 19641],
                 [19641, 19643, 19637, 19638],
                 [19638, 19643.25, 19638, 19641.25],
                 [19642.75, 19647, 19637.5, 19637.5],
                 [19639, 19643.25, 19639, 19642],
                 [19641.5, 19643, 19638.25, 19643],
                 [19643.75, 19644, 19640.75, 19643],
                 [19643.25, 19644, 19640, 19640.5],
                 [19641.5, 19644, 19640, 19644],
                 [19644.5, 19644.5, 19641.75, 19643.5],
                 [19643.25, 19644.25, 19641, 19644.25],
                 [19642.25, 19645, 19642.25, 19645],
                 [19643.25, 19643.25, 19638, 19642.5],
                 [19642.25, 19642.25, 19636.25, 19640],
                 [19640, 19640.5, 19634.5, 19636.5],
                 [19637, 19638.25, 19635.5, 19636.5],
                 [19635.25, 19636, 19632, 19633.75],
                 [19634.75, 19638, 19631, 19637],
                 [19638, 19646, 19638, 19644],
                 [19644.25, 19645, 19640, 19641],
                 [19641.5, 19643, 19641.5, 19641.5],
                 [19641.5, 19643.5, 19641.5, 19643.5],
                 [19642.75, 19644, 19640, 19644],
                 [19642.25, 19648, 19642.25, 19647],
                 [19647.5, 19649.25, 19643.5, 19646.5],
                 [19648.25, 19651.25, 19647.5, 19649.5],
                 [19647.25, 19653.75, 19647.25, 19653.25],
                 [19652, 19655, 19648.75, 19650],
                 [19649.25, 19652.25, 19647.75, 19650.5],
                 [19650.25, 19651.75, 19646.75, 19650],
                 [19651.25, 19651.5, 19648.25, 19648.25],
                 [19649, 19651.5, 19644.25, 19646.75]]
    a = BatchIndicators('year_data.dat', (4777984, 5), ob_flag=True)
    # print(a.state_space)
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
