# This is a sample Python script.
import _queue

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

import multiprocessing
from tqdm import tqdm
import time

# Worker function for each process
def worker(process_id, total_iterations, queue):
    # Simulate work by updating progress
    for i in range(total_iterations):
        time.sleep(0.001)  # Simulating work
        reward = 100 - (i*0.1)
        queue.put((process_id, i + 1, reward))  # Send progress to main process

def main():
    queue = multiprocessing.Queue()
    iterations = [1000, 2000]
    progress_bars = [tqdm(total=iterations[i], desc=f"Process {i}", position=i, ncols=100, dynamic_ncols=True) for i in range(2)]

    # Start 2 processes
    processes = []
    for i in range(2):
        p = multiprocessing.Process(target=worker, args=(i, iterations[i], queue))
        processes.append(p)
        p.start()

    # Main process updates progress bars based on messages from workers
    completed = [0, 0]
    while any(p.is_alive() for p in processes):
        try:
            process_id, progress, reward = queue.get(timeout=0.1)  # Receive progress update
            if completed[process_id] < progress:
                progress_bars[process_id].update(progress - completed[process_id])
                progress_bars[process_id].set_postfix({"Reward": f"{reward:.2f}"})
                completed[process_id] = progress
        except _queue.Empty:
            pass

    # Ensure progress bars are completed
    for i in range(2):
        progress_bars[i].close()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("\nAll processes are done!")

if __name__ == "__main__":
    main()


# if __name__ == '__main__':
    # This is for testing how the model performs on new data
    # start_training = '2011-01-03'
    # end_training = '2020-02-03'
    # env = EnviroBatchProcess(INSTRUMENT, start_training, end_training, 1, testing=False, indicator_select=INDICATOR)
    # print(env.env_out[-1][-1])
    # observation_, reward_unreal, reward_real = env.step(['buy'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal, env.orders['open'])
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['hold'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)
    # observation_, reward_unreal, reward_real = env.step(['sell'])
    # print(env.env_out[-1][-1], reward_real, reward_unreal)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
