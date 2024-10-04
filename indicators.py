
# somehow may also need to get in the economic canlendar in here. It also might be a good idea to just iterate through
# the entire training time and just pre pull the data rather than requesting and then working.
# Data should be the previous 60 days worth of data with the most recent being at the end of the list

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

        self.rsi_last_values = []
        self.mac_last_values = [[], []]  # list of 2 items, the first is list of macd and the second is list of trigger
        self.ob_up_last_values = []
        self.ob_down_last_values = []
        self.fvgs_up_last_values = []
        self.fvgs_down_last_values = []

        if self.rsi_flag:
            self.rsi_init()
        if self.mac_flag:
            self.mac_init()
        if self.ob_flag:
            self.ob_init()
        if self.fvg_flag:
            self.fvg_init()
        if self.news_flag:
            pass

    def get_indicators(self, cur_data):
        self.list_data.pop(0)
        self.list_data.append(cur_data)

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


test_data = [[19629,19645.25,19614.5,19623.5,45522.75],
[19623.25,19649.75,19623,19641.5,45522.7506944444],
[19641,19672.75,19640.75,19667.5,45522.7513888889],
[19665.5,19673.75,19660,19661.5,45522.7520833333],
[19662.25,19665.25,19653.25,19656,45522.7527777778],
[19657,19659.25,19650.25,19654,45522.7534722222],
[19653.25,19664,19652.5,19662.25,45522.7541666667],
[19662.25,19662.75,19655.5,19657,45522.7548611111],
[19657.5,19658.75,19656,19658,45522.7555555556],
[19657,19658.75,19655,19655,45522.75625],
[19655,19663,19653.25,19661.75,45522.7569444444],
[19662,19663.25,19657.25,19661.25,45522.7576388889],
[19661,19663.5,19659,19660,45522.7583333333],
[19660.25,19662.75,19642.25,19645.75,45522.7590277778],
[19646.75,19654,19646.25,19653,45522.7597222222],
[19653.5,19654,19645.75,19646.5,45522.7604166667],
[19646.75,19647,19640.25,19645,45522.7611111111],
[19645.5,19647.25,19640,19640,45522.7618055556],
[19640.25,19640.5,19633,19635.5,45522.7625],
[19638,19648.5,19637.75,19641.5,45522.7631944444],
[19641.5,19642.75,19640,19641.25,45522.7638888889],
[19641.25,19641.25,19639.25,19639.75,45522.7645833333],
[19639.75,19642.5,19635.5,19639.75,45522.7652777778],
[19639.75,19639.75,19636.75,19637.5,45522.7659722222],
[19638.75,19641,19627.5,19631.25,45522.7666666667],
[19632.5,19633,19621,19629,45522.7673611111],
[19629.25,19636.25,19627.25,19633.75,45522.7680555556],
[19635,19647,19634.25,19637.25,45522.76875],
[19636.75,19642,19633,19641,45522.7694444444],
[19641,19643,19637,19638,45522.7701388889],
[19638,19643.25,19638,19641.25,45522.7708333333],
[19642.75,19647,19637.5,19637.5,45522.7715277778],
[19639,19643.25,19639,19642,45522.7722222222],
[19641.5,19643,19638.25,19643,45522.7729166667],
[19643.75,19644,19640.75,19643,45522.7736111111],
[19643.25,19644,19640,19640.5,45522.7743055556],
[19641.5,19644,19640,19644,45522.775],
[19644.5,19644.5,19641.75,19643.5,45522.7756944444],
[19643.25,19644.25,19641,19644.25,45522.7763888889],
[19642.25,19645,19642.25,19645,45522.7770833333],
[19643.25,19643.25,19638,19642.5,45522.7777777778],
[19642.25,19642.25,19636.25,19640,45522.7784722222],
[19640,19640.5,19634.5,19636.5,45522.7791666667],
[19637,19638.25,19635.5,19636.5,45522.7798611111],
[19635.25,19636,19632,19633.75,45522.7805555556],
[19634.75,19638,19631,19637,45522.78125],
[19638,19646,19638,19644,45522.7819444444],
[19644.25,19645,19640,19641,45522.7826388889],
[19641.5,19643,19641.5,19641.5,45522.7833333333],
[19641.5,19643.5,19641.5,19643.5,45522.7840277778],
[19642.75,19644,19640,19644,45522.7847222222],
[19642.25,19648,19642.25,19647,45522.7854166667],
[19647.5,19649.25,19643.5,19646.5,45522.7861111111],
[19648.25,19651.25,19647.5,19649.5,45522.7868055556],
[19647.25,19653.75,19647.25,19653.25,45522.7875],
[19652,19655,19648.75,19650,45522.7881944444],
[19649.25,19652.25,19647.75,19650.5,45522.7888888889],
[19650.25,19651.75,19646.75,19650,45522.7895833333],
[19651.25,19651.5,19648.25,19648.25,45522.7902777778]]
next = [19649,19651.5,19644.25,19646.75,45522.7909722222]
a = Indicators(test_data, ob_flag=True)
print('up', 'down')
print(a.ob_up_last_values, a.ob_down_last_values)
print(a.get_indicators(next))
a.ob()
print(a.ob_up_last_values, a.ob_down_last_values)
next = [19648,19654.25,19647.5,19653.25]
a.get_indicators(next)
a.ob()
print(a.ob_up_last_values, a.ob_down_last_values)