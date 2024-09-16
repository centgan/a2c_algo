# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from environment import EnviroTraining
from model import Agent, AgentSep
import matplotlib.pyplot as plt
import time
import concurrent.futures as cf
from datetime import timedelta, datetime

ALPHA_ACTOR = 0.0001
ALPHA_CRITIC = 0.001
GAMMA = 0.3
ACTION_SIZE = 3
LOAD_CHECK = False
INSTRUMENT = 'NAS100_USD'


def group_util(date1, date2, n_periods):
    diff = (date2 - date1) / n_periods
    return [date1 + diff * idx for idx in range(n_periods)] + [date2]


def training(instrument, start_train, end_train, core_num):
    # print(start_train, end_train, core_num)
    env = EnviroTraining(instrument, start_train, end_train)
    # agent = Agent(alpha=0.0001, action_size=3)
    agent = AgentSep(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)

    observation = env.env_out
    balance = 0
    pre_balance = 0
    highest_balance = 0
    reward_history = []
    action_mapping = ['sell', 'hold', 'buy']
    while not env.done:
        # print(observation, core_num)
        action = agent.choose_action(observation)
        # print(action)
        # print(action_mapping[action], action)
        observation_, reward_real, reward_unreal = env.step(action_mapping[action])

        reward_history.append(reward_unreal)
        balance += reward_real
        if not LOAD_CHECK:
            agent.learn(observation, reward_unreal, observation_)
        observation = observation_

        # print(str(balance) + ' running on core number: ' + str(core_num))
        if balance != pre_balance:
            print(str(balance) + ' running on core number: ' + str(core_num))
        pre_balance = round(balance, 1)
        # print(balance)
        if balance > highest_balance and not LOAD_CHECK:
            highest_balance = balance
            agent.save_model()


if __name__ == '__main__':
    start_training = '2011-01-03'
    end_training = '2020-02-03'

    start_date = datetime.strptime(start_training, '%Y-%m-%d')
    end_date = datetime.strptime(end_training, '%Y-%m-%d')
    # 4 cores
    n = 4
    # dates separated equally for 4 cores
    res = [date.strftime("%Y-%m-%d") for date in group_util(start_date, end_date, n)]

    with cf.ProcessPoolExecutor() as executor:
        # results = []
        for idx, i in enumerate(res):
            if idx == 0:
                pass
            else:
                executor.submit(training, INSTRUMENT, res[idx-1], i, idx)

    # env = EnviroTraining('NAS100_USD', '2011-01-03', '2020-02-03')
    # # agent = Agent(alpha=0.0001, action_size=3)
    # agent = AgentSep(alpha_actor=0.0001, alpha_critic=0.001, gamma=0.3, action_size=3)
    #
    # load_checkpoint = False
    #
    # observation = env.env_out
    # balance = 0
    # pre_balance = 0
    # highest_balance = 0
    # reward_history = []
    # action_mapping = ['sell', 'hold', 'buy']
    # while not env.done:
    #     # print(observation)
    #     action = agent.choose_action(observation)
    #     # print(action)
    #     # print(action_mapping[action], action)
    #     observation_, reward_real, reward_unreal = env.step(action_mapping[action])
    #     reward_history.append(reward_real)
    #     balance += reward_real
    #     if not load_checkpoint:
    #         agent.learn(observation, reward_unreal, observation_)
    #     observation = observation_
    #
    #     if len(reward_history) == 0:
    #         print('balance')
    #     if balance != pre_balance:
    #         print(balance)
    #     pre_balance = balance
    #     # print(balance)
    #     if balance > highest_balance and not load_checkpoint:
    #         highest_balance = balance
    #         agent.save_model()
    #
    # if not load_checkpoint:
    #     plt.plot(reward_history)
    #     plt.title('Realized reward over training period')
    #     plt.savefig('realized.png')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
