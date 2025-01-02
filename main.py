# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from environment import EnviroBatchProcess
from model import Agent
import matplotlib.pyplot as plt
import time
import concurrent.futures as cf
from datetime import timedelta, datetime

ALPHA_ACTOR = 0.0001
ALPHA_CRITIC = 0.001
GAMMA = 0.6
ACTION_SIZE = 3
LOAD_CHECK = False
INSTRUMENT = 'NAS100_USD'


# def group_util(date1, date2, n_periods):
#     diff = (date2 - date1) / n_periods
#     return [date1 + diff * idx for idx in range(n_periods)] + [date2]
#
#
# def training(instrument, start_train, end_train, core_num):
#     print(start_train, end_train, core_num)
#     env = EnviroBatchProcess(instrument, start_train, end_train)
#     # agent = Agent(alpha=0.0001, action_size=3)
#     agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)
#
#     observation = env.env_out
#     balance = 0
#     pre_balance = 0
#     highest_balance = 0
#     reward_history = []
#     action_mapping = ['sell', 'hold', 'buy']
#     while not env.done:
#         print(observation, core_num)
#         actions = agent.choose_action(observation)
#         # print(action)
#         # print(action_mapping[action], action)
#         observation_, reward_real, reward_unreal = env.step(action_mapping[action] for action in actions)
#
#         reward_history.append(reward_unreal)
#         balance += reward_real
#         if not LOAD_CHECK:
#             agent.learn(observation, reward_unreal, observation_)
#         observation = observation_
#
#         # print(str(balance) + ' running on core number: ' + str(core_num))
#         if balance != pre_balance:
#             print(str(round(balance, 1)) + ' running on core number: ' + str(core_num))
#         balance = round(balance, 1)
#         pre_balance = balance
#         # print(balance)
#         if balance > highest_balance and not LOAD_CHECK:
#             highest_balance = balance
#             agent.save_model()


if __name__ == '__main__':
    start_training = '2011-01-03'
    end_training = '2020-02-03'

    start_date = datetime.strptime(start_training, '%Y-%m-%d')
    end_date = datetime.strptime(end_training, '%Y-%m-%d')

    env = EnviroBatchProcess('NAS100_USD', '2011-01-03', '2020-02-03', 256)
    # agent = Agent(alpha=0.0001, action_size=3)
    agent = Agent(alpha_actor=0.0001, alpha_critic=0.001, gamma=0.6, action_size=3)

    load_checkpoint = False

    observation = env.env_out
    balance = 0
    pre_balance = 0
    highest_balance = 0
    action_mapping = ['sell', 'hold', 'buy']
    while not env.done:
        actions = agent.choose_action(observation)
        observation_, reward_real = env.step(action_mapping[action] for action in actions)

        if not load_checkpoint:
            agent.learn(observation, reward_real, observation_)
        observation = observation_

        if env.balance != pre_balance:
            print(round(env.balance, 2), round(env.year_time_step / len(env.year_data) * 100, 2))
        pre_balance = env.balance
        if env.balance > highest_balance and not load_checkpoint:
            env.balance = balance
            agent.save_model()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
