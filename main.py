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

ALPHA_ACTOR = 0.0001
ALPHA_CRITIC = 0.001
GAMMA = 0.6
ACTION_SIZE = 3
LOAD_CHECK = False
INSTRUMENT = 'NAS100_USD'


if __name__ == '__main__':
    start_training = '2011-01-03'
    end_training = '2020-02-03'

    start_date = datetime.strptime(start_training, '%Y-%m-%d')
    end_date = datetime.strptime(end_training, '%Y-%m-%d')

    env = EnviroBatchProcess('NAS100_USD', '2011-01-03', '2020-02-03', 256)
    agent1 = Agent(alpha_actor=0.0001, alpha_critic=0.001, gamma=0.6, action_size=3)
    agent2 = Agent(alpha_actor=0.0001, alpha_critic=0.001, gamma=0.6, action_size=3)
    agent3 = Agent(alpha_actor=0.0001, alpha_critic=0.001, gamma=0.6, action_size=3)
    agents = [agent1, agent2, agent3]

    load_checkpoint = False

    observation = env.env_out
    balance = 0
    pre_balance = 0
    highest_balance = 0
    action_mapping = ['sell', 'hold', 'buy']
    while not env.done:
        actions = []
        for agent in agents:
            batch_action = agent.choose_action(observation)
            actions.append(action_mapping[batch_single_action] for batch_single_action in batch_action)
        # actions = agent.choose_action(observation)
        # print(actions)
        observation_, reward_real = env.step(actions, [agent.balance for agent in agents])

        if not load_checkpoint:
            for agent_index, agent in enumerate(agents):
                agent.learn(observation, reward_real[agent_index], observation_)
                agent.update_balance(reward_real[agent_index][-1])
                # if agent.balance != pre_balance:
                print(round(agent.balance, 2), round(env.year_time_step / env.year_data_shape[0] * 100, 5), agent_index)
                pre_balance = agent.balance
                if agent.balance > highest_balance and not load_checkpoint:
                    agent.balance = balance
                    agent.save_model()
        observation = observation_


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
