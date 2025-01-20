# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from environment import EnviroBatchProcess
from model import Agent
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from tqdm import tqdm
import json
import os

ALPHA_ACTOR = 0.00001
ALPHA_CRITIC = 0.00001
GAMMA = 0.7
ACTION_SIZE = 3
LOAD_CHECK = False
INSTRUMENT = 'NAS100_USD'
EPOCHES = 2
BATCH_SIZE = 256
# below is typical retail
# INDICATORS = [1, 1, 0, 0, 1]  # in order of rsi, macd, ob, fvg, news
# below is ict
INDICATORS = [0, 0, 1, 1, 1]

if __name__ == '__main__':
    start_training = '2011-01-03'
    end_training = '2020-02-03'

    start_date = datetime.strptime(start_training, '%Y-%m-%d')
    end_date = datetime.strptime(end_training, '%Y-%m-%d')

    # needed for pytorch as it requires the input size so this calculates that
    input_size = 5  # 60 past candles * 5 features (OHLC Date)
    input_size += sum(INDICATORS[:2]) + INDICATORS[-1]  # rsi, macd and news all only add 1
    input_size += 10 if INDICATORS[2] else 0  # 10 additional parameters for ob
    input_size += 20 if INDICATORS[3] else 0  # 20 additional parameters for fvg
    if os.path.exists('./results'):
        os.mkdir('./results')

    for gamma_val in range(0, 105, 5):
        agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=gamma_val, action_size=ACTION_SIZE, input_size=input_size)
        env = EnviroBatchProcess(INSTRUMENT, '2011-01-03', '2020-02-03', BATCH_SIZE, indicator_select=INDICATORS)

        observation = env.env_out
        pre_balance = 0
        balance_history = []
        highest_balance = 0
        action_mapping = ['sell', 'hold', 'buy']
        with tqdm(total=env.year_data_shape[0], desc=f'Gamma value {gamma_val}', ncols=100) as pbar:
            while not env.done:
                pbar.set_postfix({"Reward": f"{env.balance:.2f}"})
                actions = agent.choose_action(observation)
                # print(actions)
                observation_, reward_real = env.step(action_mapping[action] for action in actions)
                if observation_.size == 0:
                    continue

                if not LOAD_CHECK:
                    agent.learn(observation, reward_real, observation_)

                observation = observation_
                # if env.balance != pre_balance:
                balance_history.append(env.balance)
                pre_balance = env.balance
                if env.balance > highest_balance and not LOAD_CHECK:
                    highest_balance = env.balance
                    agent.save_model()

                pbar.update(BATCH_SIZE)

            with open(f'./results/{datetime.now().strftime("%Y-%m-%d_%H:%M")}_{gamma_val}.json', 'w') as f:
                json.dump(balance_history, f)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
