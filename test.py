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
INDICATOR = [0, 0, 0, 0, 0]  # same here rsi, mac, ob, fvg, news

if __name__ == '__main__':
    # This is for testing how the model performs on new data
    start_training = '2011-01-03'
    end_training = '2020-02-03'
    env = EnviroBatchProcess(INSTRUMENT, start_training, end_training, 1, testing=False, indicator_select=INDICATOR)
    print(env.env_out[-1][-1])
    observation_, reward_unreal, reward_real = env.step(['buy'])
    print(env.env_out[-1][-1], reward_real, reward_unreal, env.orders['open'])
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['hold'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)
    observation_, reward_unreal, reward_real = env.step(['sell'])
    print(env.env_out[-1][-1], reward_real, reward_unreal)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
