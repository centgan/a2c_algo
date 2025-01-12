# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from environment import EnviroBatchProcess
from model import Agent
import matplotlib.pyplot as plt
import time
import concurrent.futures as cf
from datetime import timedelta, datetime
import numpy as np

ALPHA_ACTOR = 0.00001
ALPHA_CRITIC = 0.00001
GAMMA = 0.7
ACTION_SIZE = 3
LOAD_CHECK = False
INSTRUMENT = 'NAS100_USD'
EPOCHES = 2
# below is typical retail
# INDICATORS = [1, 1, 0, 0, 1]  # in order of rsi, macd, ob, fvg, news
# below is ict
INDICATORS = [0, 0, 1, 1, 1]

if __name__ == '__main__':
    start_training = '2011-01-03'
    end_training = '2020-02-03'

    start_date = datetime.strptime(start_training, '%Y-%m-%d')
    end_date = datetime.strptime(end_training, '%Y-%m-%d')

    input_size = 5  # 60 past candles * 5 features (OHLC Date)
    input_size += sum(INDICATORS[:2]) + INDICATORS[-1]  # rsi, macd and news all only add 1
    input_size += 10 if INDICATORS[2] else 0  # 10 additional parameters for ob
    input_size += 20 if INDICATORS[3] else 0  # 20 additional parameters for fvg

    agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE, input_size=input_size)
    for epoch in range(EPOCHES):
        env = EnviroBatchProcess(INSTRUMENT, '2011-01-03', '2020-02-03', 256, indicator_select=INDICATORS)

        observation = env.env_out
        balance = 0
        pre_balance = 0
        highest_balance = 0
        action_mapping = ['sell', 'hold', 'buy']
        while not env.done:
            actions = agent.choose_action(observation)
            # print(actions)
            observation_, reward_real = env.step(action_mapping[action] for action in actions)
            if observation_.size == 0:
                continue

            if not LOAD_CHECK:
                agent.learn(observation, reward_real, observation_)

            observation = observation_
            # if env.balance != pre_balance:
            print(round(env.balance, 2), round(env.year_time_step / env.year_data_shape[0] * 100, 5))
            pre_balance = env.balance
            if env.balance > highest_balance and not LOAD_CHECK:
                highest_balance = env.balance
                agent.save_model()
        print(f'epoch #{epoch} finished running current balance is {env.balance}')
    print('training complete final reward: ', env.balance)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
