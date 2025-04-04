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
import numpy as np
import multiprocessing
import _queue
import time

ALPHA_ACTOR = 0.0005
ALPHA_CRITIC = 0.0007
GAMMA = 0.7
ACTION_SIZE = 3
LOAD_CHECK = False
INSTRUMENT = 'NAS100_USD'
EPOCHES = 2
BATCH_SIZES = [1, 4, 16, 32, 128, 256]
BATCH_SIZE = 256
LAMBDA = 0.8
# below is typical retail
INDICATORS = [1, 1, 0, 0, 1]  # in order of rsi, macd, ob, fvg, news
# below is ict
# INDICATORS = [0, 0, 1, 1, 1]

NUM_AGENTS = 4
START_TRAINING = datetime.strptime('2011-01-03', '%Y-%m-%d')
END_TRAINING = datetime.strptime('2020-02-03', '%Y-%m-%d')

def determine_batch_size(percentage):
    if percentage < 0.4:
        return BATCH_SIZES[0]
    elif percentage < 5:
        return BATCH_SIZES[1]
    elif percentage < 10:
        return BATCH_SIZES[2]
    elif percentage < 30:
        return BATCH_SIZES[3]
    elif percentage < 60:
        return BATCH_SIZES[4]
    else:
        return BATCH_SIZES[5]

# agent_id is 0 based
def agent_worker(agent_id, global_memory_, lock_, queue_):
    os.makedirs(f'./results/{agent_id}_agent', exist_ok=True)

    date_delta = (END_TRAINING - START_TRAINING).days // NUM_AGENTS
    agent_start = START_TRAINING + timedelta(agent_id * date_delta)

    # loop the environment start dates to ensure that start time steps are the same for all workers
    looping = [[agent_start, END_TRAINING]]
    if agent_start != START_TRAINING:
        looping.append([START_TRAINING, agent_start])

    steps = 0
    while steps < 1_000_000:
        steps += 3 * (agent_id + 1)
        queue_.put((agent_id, steps))
    # for loop in looping:
    #     agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)
    #     env = EnviroBatchProcess(INSTRUMENT, loop[0].strftime("%Y-%m-%d"), loop[1].strftime("%Y-%m-%d"), 1, indicator_select=INDICATORS)
    #     while not env.done:

def learner(global_memory_, lock_):
    agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)

    while True:
        if len(global_memory_) >= 32:  # Wait until we have enough experiences
            with lock_:
                batch = global_memory_[:32]
                del global_memory_[:32]  # Remove used experiences

            # Convert batch to tensors
            states, actions, rewards, next_states = zip(*batch)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)

            # Perform batch update
            agent.learn(states, rewards, next_states)

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    queue = multiprocessing.Queue()
    global_memory = manager.list()
    lock = manager.Lock()
    progress_bars = [tqdm(total=1_000_000, desc=f"Process {i}", position=i, ncols=100, dynamic_ncols=True) for i in range(NUM_AGENTS)]

    processes = []
    for i in range(NUM_AGENTS):
        p = multiprocessing.Process(target=agent_worker, args=(i, global_memory, lock, queue))
        processes.append(p)
        p.start()

    learner_process = multiprocessing.Process(target=learner, args=(global_memory, lock))
    learner_process.start()

    completed = [0] * NUM_AGENTS
    while any(p.is_alive() for p in processes):
        try:
            process_id, progress = queue.get(timeout=0.1)
            if completed[process_id] < progress:
                progress_bars[process_id].update(progress - completed[process_id])
                completed[process_id] = progress
        except _queue.Empty:
            pass

    for i in progress_bars:
        i.close()
    for p in processes:
        p.join()

    # agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)
    # if not os.path.exists('./results'):
    #     os.mkdir('./results')
    #
    # for epoch in range(EPOCHES):
    #     env = EnviroBatchProcess(INSTRUMENT, '2011-01-03', '2020-02-03', BATCH_SIZES[0], indicator_select=INDICATORS)
    #
    #     balance_history = []
    #     pre_balance = 0
    #     highest_balance = 0
    #     action_mapping = ['sell', 'hold', 'buy']
    #
    #     with tqdm(total=env.year_data_shape[0], desc=f'Epoch {epoch + 1}/{EPOCHES}', ncols=100) as pbar:
    #         while not env.done:
    #             env.batch_size = determine_batch_size((env.year_time_step / env.year_data_shape[0]) * 100)
    #             # print(env.batch_size)
    #             observation = env.env_out
    #             pbar.set_postfix({"Reward": f"{env.balance:.2f}"})
    #             actions = agent.choose_action(observation)
    #             # print(actions)
    #             actions_mapped = [action_mapping[action] for action in actions]
    #             observation_, reward_unreal, reward_real = env.step(actions_mapped)
    #             print(reward_unreal, reward_real, actions)
    #             reward_real_lambda = np.multiply(LAMBDA, reward_real)
    #             reward_unreal_lambda = np.multiply((1 - LAMBDA), reward_unreal)
    #             training_reward = np.add(reward_real_lambda, reward_unreal_lambda).tolist()
    #
    #             if observation_.size == 0:
    #                 continue
    #
    #             if not LOAD_CHECK:
    #                 agent.learn(observation, training_reward, observation_)
    #
    #             observation = observation_
    #             balance_history.append(env.balance)
    #             # print(round(env.balance, 2), round(env.year_time_step / env.year_data_shape[0] * 100, 5))
    #             # print(reward_unreal, reward_real, actions)
    #             pre_balance = env.balance
    #             if env.balance > highest_balance and not LOAD_CHECK:
    #                 highest_balance = env.balance
    #                 agent.save_model()
    #
    #             pbar.update(env.batch_size)
    #
    #         with open(f'./results/{datetime.now().strftime("%Y-%m-%d_%H:%M")}_{epoch}.json', 'w') as f:
    #             json.dump(balance_history, f)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
