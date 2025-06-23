import pandas as pd
from datetime import datetime, timedelta
import pytz
import src.indicators as indicators
import numpy as np
import pyarrow.parquet as pq

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
        self.current_year = self.train_start.year
        self.batch_size = batch_size
        self.indicator_select = indicator_select

        self.path_to_data = './data/test_data/full_training_data_' if testing else './data/full_training_data_'
        self.path_to_indicator = './indicators/test/' if testing else './indicators/train/'

        self.start_opening = 0  # this is opening price for compression and decompression
        self.first_date = datetime.now()  # just as a placeholder for opening date for compression and decompression

        self.orders = {"open": [], "closed": []}
        # self.year_time_step = 60  # keeps track of which index in the year currently on
        self.chunk_time_step = 60
        self.balance = 0
        self.done = False
        self.commission = 0.5

        # self.year_data_filename = 'train_year_data.dat' if not testing else 'test_year_data.dat'
        self.chunk_gen = self.fetch_chunk_data()
        _, self.chunk_data = next(self.chunk_gen)
        self.chunk_indicator = None
        self.batch_data_shape = self.chunk_data.shape
        self.indicator_class = indicators.BatchJITIndicator(self.chunk_data, indicator_select, testing=testing)
        self.env_out = self.get_env_state()

    def fetch_chunk_data(self):
        pf = pq.ParquetFile(self.path_to_data + str(self.current_year) + '.parquet')
        window_size = timedelta(days=30)
        target_start = self.train_start
        buffer = []

        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i)
            df = table.to_pandas()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # Feed rows into appropriate window(s)
            for _, row in df.iterrows():
                ts = row['timestamp']
                if target_start <= ts < target_start + window_size:
                    buffer.append(row)
                elif ts >= target_start + window_size:
                    # Yield completed window
                    df_buffer = pd.DataFrame(buffer)
                    df_buffer['timestamp'] = df_buffer['timestamp'].astype('int64') // 10 ** 9
                    yield target_start, df_buffer.to_numpy()
                    buffer = []
                    target_start += window_size

                    # Check if current row belongs in next window
                    if target_start <= ts < target_start + window_size:
                        buffer.append(row)

        if buffer:
            df_buffer = pd.DataFrame(buffer)
            df_buffer['timestamp'] = df_buffer['timestamp'].astype('int64') // 10 ** 9
            yield target_start, df_buffer.to_numpy()

    def get_env_state(self):
        returning_batch = []
        if self.chunk_time_step + self.batch_size > self.batch_data_shape[0]:
            remainder = self.batch_size - self.chunk_time_step
            copied_chuck_end = self.chunk_data[-60-remainder:].copy()
            try:
                # print('in')
                _, self.chunk_data = next(self.chunk_gen)
                self.chunk_data = np.concatenate((copied_chuck_end, self.chunk_data), axis=0)
                self.batch_data_shape = self.chunk_data.shape
                self.chunk_time_step = 60
                # print(self.chunk_data.shape)
                self.indicator_class.process(self.chunk_data, remainder)

            except StopIteration:
                self.current_year += 1
                if int(self.current_year) > 2020:
                    self.done = True
                    return np.array([])
                else:
                    self.chunk_gen = self.fetch_chunk_data()

                    _, self.chunk_data = next(self.chunk_gen)
                    self.chunk_data = np.append(self.chunk_data, copied_chuck_end)
                    self.batch_data_shape = self.chunk_data.shape
                    self.chunk_time_step = 60
                    self.indicator_class.process(self.chunk_data, remainder)


        end_index = self.chunk_time_step + self.batch_size

        indicator_names = ['rsi', 'macd', 'ob', 'fvg', 'news']
        for i in range(self.chunk_time_step, end_index):
            individual_batch = np.array(self.chunk_data[i-60:i].copy())

            # print(individual_batch.shape, i, self.chunk_time_step, end_index, 'individual', self.chunk_data.shape, self.batch_data_shape)
            for indicator_name in indicator_names:
                if indicator_name in self.indicator_class.indicator_out.keys():
                    # print(np.array(self.indicator_class.indicator_out[indicator_name][i-60:i]).shape, indicator_name)
                    individual_batch = np.column_stack((individual_batch, self.indicator_class.indicator_out[indicator_name][i-60:i]))

            returning_batch.append(individual_batch)
        return np.array(returning_batch)

    # actions is a list of 256
    def step(self, actions):
        # fetch the data for the next year
        # year_data = np.memmap(self.year_data_filename, dtype=np.float64, mode='r', shape=self.year_data_shape)
        # if self.year_time_step >= self.year_data_shape[0]:
        #     self.done = True
        #
        #     # once after running through the entire year's data will write all the orders to a file
        #     with open('orders.json', 'w') as write:
        #         json.dump(self.orders, write)

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
                        'entry_datetime': self.chunk_data[self.chunk_time_step+action_index][-1],
                        'exit_datetime': '',
                        'entry_price': self.chunk_data[self.chunk_time_step+action_index][-2],
                        'exit_price': '',
                        'order': action
                    })
                    self.balance -= self.commission  # commission for now is 1
                    returning_reward.append(0)
                else:
                    if (action == 'buy' and self.orders['open'][0]['order'] == 'sell') or (action == 'sell' and self.orders['open'][0]['order'] == 'buy'):
                        move_to_close = self.orders['open'].pop()
                        move_to_close['exit_datetime'] = self.chunk_data[self.chunk_time_step+action_index][-1]
                        move_to_close['exit_price'] = self.chunk_data[self.chunk_time_step+action_index][-2]
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
                        current_close_price = self.chunk_data[self.chunk_time_step + action_index][-2]
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
                    current_close_price = self.chunk_data[self.chunk_time_step + action_index][-2]
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
                move_to_close['exit_datetime'] = self.chunk_data[self.chunk_time_step + action_index][-1]
                move_to_close['exit_price'] = self.chunk_data[self.chunk_time_step + action_index][-2]
                self.orders['closed'].append(move_to_close)
                self.balance += returning_reward[-1] - self.commission
                returning_reward[-1] = 0

            # max time limit, trades held at 6pm est will be auto liquidated
            if (datetime.fromtimestamp(int(self.chunk_data[self.chunk_time_step + action_index][-1])).hour == 18 and
                    datetime.fromtimestamp(int(self.chunk_data[self.chunk_time_step + action_index][-1])).minute == 0 and
                    len(self.orders['open']) != 0):
                move_to_close = self.orders['open'].pop()
                move_to_close['exit_datetime'] = self.chunk_data[self.chunk_time_step + action_index][-1]
                move_to_close['exit_price'] = self.chunk_data[self.chunk_time_step + action_index][-2]
                self.orders['closed'].append(move_to_close)
                self.balance += returning_reward[-1] - self.commission
                returning_reward[-1] = 0

            returning_realized.append(self.balance)

        # print(returning_reward)
        # updating state space to get next batch ie [:256] => [256:512]
        self.chunk_time_step += self.batch_size
        self.env_out = self.get_env_state()

        return [self.env_out, returning_reward, returning_realized]


if __name__ == '__main__':
    train_start = '2011-01-03'
    train_end = '2020-02-03'
    # # train_start = datetime.strptime(train_start, "%Y-%m-%d %H:%M:%S")
    # # train_final_end = datetime.strptime(train_end, "%Y-%m-%d %H:%M:%S")
    # # train_end = train_start
    instrument = 'NAS100_USD'
    a = EnviroBatchProcess(instrument, train_start, train_end, 1, [0, 0, 0, 0, 0])
    print(a.env_out)
    # for win_start, chunk in a.fetch_chunk_data():
    #     print(f"Chunk starting at {win_start}: {len(chunk)} rows")

    # for i in a.year_indicators[0]:
    #     print(i)
    # for i in a.year_data:
    #     print(i)
    # 2011-12-30 16:14:00, Current: 2012-01-03 06:00:00
    # print(a.fetch_current_year_data(2020))
    # print(a.year_data)
