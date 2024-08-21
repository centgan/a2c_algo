# This is a sample Python script.
from datetime import timedelta

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from environment import EnviroTraining
from model import Agent, AgentSep
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = EnviroTraining('NAS100_USD', '2011-01-03', '2020-02-01')
    # agent = Agent(alpha=0.0001, action_size=3)
    agent = AgentSep(alpha_actor=0.0001, alpha_critic=0.001, gamma=0.3, action_size=3)

    load_checkpoint = False

    observation = env.env_out
    balance = 0
    highest_balance = 0
    reward_history = []
    action_mapping = ['sell', 'hold', 'buy']
    while not env.done:
        # print(observation)
        action = agent.choose_action(observation)
        # print(action)
        # print(action_mapping[action], action)
        observation_, reward_real, reward_unreal = env.step(action_mapping[action])
        reward_history.append(reward_real)
        balance += reward_real
        if not load_checkpoint:
            agent.learn(observation, reward_unreal, observation_)
        observation = observation_

        print(balance)
        if balance > highest_balance and not load_checkpoint:
            highest_balance = balance
            agent.save_model()

    if not load_checkpoint:
        plt.plot(reward_history)
        plt.title('Realized reward over training period')
        plt.savefig('realized.png')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
