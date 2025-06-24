# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from environment import EnviroBatchProcess
from model import Agent
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import json
import requests



ALPHA_ACTOR = 0.00001
ALPHA_CRITIC = 0.00001
GAMMA = 0.7
ACTION_SIZE = 3
LOAD_CHECK = False
INSTRUMENT = 'NAS100_USD'
INDICATOR = [1, 1, 0, 0, 1]  # same here rsi, mac, ob, fvg, news

api = "api-fxpractice.oanda.com"
account_id = "101-002-25776676-003"
token = "d8e783a23ff8bab21476e440b3d578ef-207d5f7676d7d82839a50d4907b3d6e6"


def test_main():
    # This is for testing how the model performs on new data
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
        print(round(env.balance, 2), round(env.year_time_step / env.batch_data_shape[0] * 100, 5))
        overall_balance.append(env.balance)
        pre_balance = env.balance

    print('Testing has been complete final reward: ', env.balance)
    fig, ax = plt.subplots()
    ax.plot(overall_balance)

    ax.set(xlabel='Date index', ylabel='Price ($))',
           title='Balance during testing')
    ax.grid()

    fig.savefig("testing.png")

def oanda_fetch(from_, to_):
    train_start = datetime.strptime(from_, "%Y-%m-%d %H:%M:%S")
    train_end = datetime.strptime(to_, "%Y-%m-%d %H:%M:%S")

    instrument = 'NAS100_USD'
    header = {'Authorization': 'Bearer ' + token}
    hist_path = f'/v3/accounts/{account_id}/instruments/' + instrument + '/candles'
    full_data = {
        'o': [],
        'h': [],
        'l': [],
        'c': [],
        'timestamp': []
    }

    from_time = time.mktime(pd.to_datetime(train_start).timetuple())
    to_time = time.mktime(pd.to_datetime(train_end).timetuple())

    query = {'from': str(from_time), 'to': str(to_time), 'granularity': 'M1'}
    try:
        response = requests.get('https://' + api + hist_path, headers=header, params=query)
    except:
        full_data.extend([train_start, train_end, 'failed on this data pull'])
        with open('full_training_data.json', 'w') as write:
            json.dump(full_data, write)
        time.sleep(10)
        response = requests.get('https://' + api + hist_path, headers=header, params=query)
    into_json = response.json()
    print(into_json)
    candles = []
    for candle_index, candle in enumerate(into_json['candles']):
        print(candle)

if __name__ == '__main__':
    with open('array.json', 'r') as read:
        a = json.load(read)
    plt.plot(a['timestamp'])
    plt.show()
    # oanda_fetch('2011-01-23 16:00:00', '2011-01-25 11:00:00')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
