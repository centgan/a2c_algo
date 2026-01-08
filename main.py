# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
from src.environment import EnviroBatchProcess
from src.model import Agent
from datetime import timedelta, datetime
from tqdm import tqdm
import numpy as np
import multiprocessing
import _queue
from loguru import logger
import tensorflow as tf
import json


ALPHA_ACTOR = 0.0005
ALPHA_CRITIC = 0.0007
GAMMA = 0.95
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
ACTION_MAPPING = ['sell', 'hold', 'buy']

NUM_AGENTS = 4
START_TRAINING = datetime.strptime('2011-01-03', '%Y-%m-%d')
END_TRAINING = datetime.strptime('2020-02-03', '%Y-%m-%d')

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Configure TensorFlow to use memory growth to prevent OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally limit memory per process (e.g., 4GB per agent)
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        # )
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

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
    logger.remove()
    logger.add(
        f"./logs/{agent_id}_log.log",
        format="{time} {level} {message}",
        level="DEBUG",
        rotation="5 MB",
        retention="5 days",
        enqueue=True,  # Required for multiprocessing safety
        backtrace=True,
        diagnose=True
    )
    path = f'./results/{agent_id}'
    os.makedirs(path, exist_ok=True)

    date_delta = (END_TRAINING - START_TRAINING).days // NUM_AGENTS
    agent_start = START_TRAINING + timedelta(agent_id * date_delta)

    # loop the environment start dates to ensure that start time steps are the same for all workers
    time_frame_looping = [[agent_start, END_TRAINING]]
    if agent_start != START_TRAINING:
        time_frame_looping.append([START_TRAINING, agent_start])

    last_weight_updated = 0.0
    # steps = 0
    # while steps < 1_000_000:
    #     steps += 3 * (agent_id + 1)
    #     queue_.put((agent_id, steps))
    for loop in time_frame_looping:
        agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)
        env = EnviroBatchProcess(INSTRUMENT, loop[0].strftime("%Y-%m-%d"), loop[1].strftime("%Y-%m-%d"), 1, indicator_select=INDICATORS)
        # print('starting loop: ', loop)
        logger.info(f"Starting loop: {loop}")
        
        # Track performance metrics for this agent
        performance_metrics = {
            'balance_history': [],
            'reward_unreal_history': [],
            'reward_real_history': [],
            'timestamps': [],
            'num_trades': 0
        }
        
        while not env.done:
            observation = env.env_out
            # if agent_id == 2:
            #     print(observation)
            #     print(observation.shape)
            actions = agent.choose_action(observation)
            # print(actions)
            actions_mapped = [ACTION_MAPPING[action] for action in actions]
            
            try:
                observation_, reward_unreal, reward_real = env.step(actions_mapped)
            except Exception as e:
                print('Error: {}'.format(e))
                quit()
            # print(agent_id, reward_unreal, reward_real)
            logger.info(f"Reward unrealized: {reward_unreal}, Real realized: {reward_real}")
            
            # Track metrics
            performance_metrics['balance_history'].append(float(env.balance))
            performance_metrics['reward_unreal_history'].append([float(r) for r in reward_unreal])
            performance_metrics['reward_real_history'].append([float(r) for r in reward_real])
            performance_metrics['timestamps'].append(int(env.chunk_data[-1][-1]))
            performance_metrics['num_trades'] = len(env.orders['closed'])

            with lock_:
                # Normalize rewards separately to handle different scales
                reward_unreal_arr = np.array(reward_unreal, dtype=np.float32)
                reward_real_arr = np.array(reward_real, dtype=np.float32)
                
                # Standardize each reward type separately
                unreal_mean = np.mean(reward_unreal_arr)
                unreal_std = np.std(reward_unreal_arr) + 1e-8
                normalized_unreal = (reward_unreal_arr - unreal_mean) / unreal_std
                
                real_mean = np.mean(reward_real_arr)
                real_std = np.std(reward_real_arr) + 1e-8
                normalized_real = (reward_real_arr - real_mean) / real_std
                
                # Combine normalized rewards with weighting
                balanced_reward = (0.3 * normalized_unreal) + (0.7 * normalized_real)
                
                # Clip to prevent extreme values
                balanced_reward = np.clip(balanced_reward, -10.0, 10.0)
                
                global_memory_.append((observation, actions, balanced_reward, observation_))

            # only need to check 1 of the 2 filepaths because they should always be updating together
            thing = agent.critic.sync_dir + '/critic.npy'
            if os.path.isfile(thing):
                if os.path.getmtime(thing) != last_weight_updated:
                    # print('learned')
                    logger.info('Learned!')
                    agent.load_sync_model(INDICATORS)
                    last_weight_updated = os.path.getmtime(thing)
                    # print('loaded')
            
            if agent_id == 0:
                print(f"[{agent_id}]: {env.chunk_time_step}, {datetime.fromtimestamp(env.chunk_data[-1][-1])}, {env.balance}")
        
        # Save final metrics when loop completes (only save once at the end to avoid I/O overhead)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_metrics_file = f'{path}/final_metrics_{timestamp_str}.json'
        with open(final_metrics_file, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        logger.info(f"Saved final metrics to {final_metrics_file}")
        print(f"[Agent {agent_id}] Saved {len(performance_metrics['balance_history'])} steps to {final_metrics_file}")

def learner(global_memory_, lock_):
    # this is the global agent the one that receives all the training a
    agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA, action_size=ACTION_SIZE)
    batch_size = 64
    
    # Checkpoint tracking
    update_count = 0
    best_avg_reward = float('-inf')  # Start at negative infinity to allow negative rewards
    reward_history = []
    checkpoint_interval = 1000  # Save checkpoint every 100 updates
    
    while True:
        # print(len(global_memory_), len(global_memory_) >= 64)
        if len(global_memory_) >= batch_size:  # Wait until we have enough experiences
            # print('mem is full')
            with lock_:
                batch = global_memory_[:batch_size]
                del global_memory_[:batch_size]  # Remove used experiences

            # Extract rewards from batch for tracking
            _, _, rewards, _ = zip(*batch)
            avg_reward = np.mean([np.mean(r) for r in rewards])
            reward_history.append(avg_reward)
            
            # Perform batch update
            agent.batch_learn(batch)
            agent.save_sync_model()  # Always save for agent synchronization
            
            update_count += 1
            
            # Periodic checkpoint saving (every 100 updates)
            if update_count % checkpoint_interval == 0:
                # Calculate rolling average over last 100 updates
                recent_avg = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
                
                print(f"[Learner] Update {update_count} | Avg Reward (last batch): {avg_reward:.4f} | "
                      f"Rolling Avg (1000): {recent_avg:.4f} | Best: {best_avg_reward:.4f}")
                
                # Save checkpoint
                agent.save_model()
                
                # If this is the best model so far, save it separately
                if recent_avg > best_avg_reward:
                    best_avg_reward = recent_avg
                    print(f"[Learner] New best model! Avg reward: {best_avg_reward:.4f}")
                    agent.save_best_model()  # Save to separate best_model directory

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    queue = multiprocessing.Queue()
    global_memory = manager.list()
    lock = manager.Lock()
    # progress_bars = [tqdm(total=4777984, desc=f"Process {i}", position=i, ncols=100, dynamic_ncols=True) for i in range(NUM_AGENTS)]

    processes = []
    for i in range(NUM_AGENTS):
        p = multiprocessing.Process(target=agent_worker, args=(i, global_memory, lock, queue))
        processes.append(p)
        p.start()

    learner_process = multiprocessing.Process(target=learner, args=(global_memory, lock))
    learner_process.start()

    # completed = [0] * NUM_AGENTS
    # prog_comp = [0] * NUM_AGENTS
    # while any(p.is_alive() for p in processes):
    #     try:
    #         process_id, progress, reward = queue.get(timeout=0.1)
    #         if completed[process_id] < progress:
    #             print(f"[{process_id}]: {prog_comp[process_id]}, {reward}")
    #             # progress_bars[process_id].update(progress - completed[process_id])
    #             # progress_bars[process_id].set_postfix({"Reward": f"{reward:.2f}"})
    #             completed[process_id] = progress
    #             prog_comp[process_id] += progress
    #     except _queue.Empty:
    #         pass

    # for i in progress_bars:
    #     i.close()
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
    #
    #     with tqdm(total=env.year_data_shape[0], desc=f'Epoch {epoch + 1}/{EPOCHES}', ncols=100) as pbar:
    #         while not env.done:
    #             env.batch_size = determine_batch_size((env.year_time_step / env.year_data_shape[0]) * 100)
    #             # print(env.batch_size)
    #             observation = env.env_out
    #             pbar.set_postfix({"Reward": f"{env.balance:.2f}"})
    #             actions = agent.choose_action(observation)
    #             # print(actions)
    #             actions_mapped = [ACTION_MAPPING[action] for action in actions]
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
