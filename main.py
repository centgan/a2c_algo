# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from environment import EnviroBatchProcess
from model import Agent
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import os
from tqdm import tqdm
import json

ALPHA_ACTOR = 0.0005
ALPHA_CRITIC = 0.0007
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

    agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)
    if not os.path.exists('./results'):
        os.mkdir('./results')

    for epoch in range(EPOCHES):
        env = EnviroBatchProcess(INSTRUMENT, '2011-01-03', '2020-02-03', BATCH_SIZE, indicator_select=INDICATORS)

        observation = env.env_out
        balance_history = []
        pre_balance = 0
        highest_balance = 0
        action_mapping = ['sell', 'hold', 'buy']

        with tqdm(total=env.year_data_shape[0], desc=f'Epoch {epoch + 1}/{EPOCHES}', ncols=100) as pbar:
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
                balance_history.append(env.balance)
                # print(round(env.balance, 2), round(env.year_time_step / env.year_data_shape[0] * 100, 5))
                pre_balance = env.balance
                if env.balance > highest_balance and not LOAD_CHECK:
                    highest_balance = env.balance
                    agent.save_model()

                pbar.update(BATCH_SIZE)

            with open(f'./results/{datetime.now().strftime("%Y-%m-%d_%H:%M")}_{epoch}.json', 'w') as f:
                json.dump(balance_history, f)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
