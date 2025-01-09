# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from environment import EnviroBatchProcess
from model import Agent
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

ALPHA_ACTOR = 0.00001
ALPHA_CRITIC = 0.00001
GAMMA = 0.7
ACTION_SIZE = 3
LOAD_CHECK = False
INSTRUMENT = 'NAS100_USD'
INDICATOR = [1, 1, 0, 0, 1]  # same here rsi, mac, ob, fvg, news

if __name__ == '__main__':
    start_training = '2020-02-04'
    end_training = '2024-12-31'

    start_date = datetime.strptime(start_training, '%Y-%m-%d')
    end_date = datetime.strptime(end_training, '%Y-%m-%d')

    agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)

    if LOAD_CHECK:
        agent.load_model()

    env = EnviroBatchProcess(INSTRUMENT, start_training, end_training, 256, testing=True, indicator_select=INDICATOR)
    observation = env.env_out
    balance = 0
    pre_balance = 0
    highest_balance = 0
    action_mapping = ['sell', 'hold', 'buy']
    overall_balance = []
    while not env.done:
        actions = agent.choose_action(observation)
        observation_, reward_real = env.step(action_mapping[action] for action in actions)
        if observation_.size == 0:
            continue

        observation = observation_
        print(round(env.balance, 2), round(env.year_time_step / env.year_data_shape[0] * 100, 5))
        overall_balance.append(env.balance)
        pre_balance = env.balance

    print('Testing has been complete final reward: ', env.balance)
    fig, ax = plt.subplots()
    ax.plot(overall_balance)

    ax.set(xlabel='Date index', ylabel='Price ($))',
           title='Balance during testing')
    ax.grid()

    fig.savefig("testing.png")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
