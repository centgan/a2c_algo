# A2C Multi-Agent Training with Single/Multi-GPU support
# Usage: python main.py --help

import os
import argparse
from src.environment import EnviroBatchProcess
from src.model import Agent
from src.checkpoint_manager import CheckpointManager
from datetime import timedelta, datetime
from tqdm import tqdm
import numpy as np
import multiprocessing
import _queue
from loguru import logger
import tensorflow as tf
import json
import signal
import sys


# ─── Default Hyperparameters ─────────────────────────────────────────────────
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

START_TRAINING = datetime.strptime('2011-01-03', '%Y-%m-%d')
END_TRAINING = datetime.strptime('2020-02-03', '%Y-%m-%d')

# Checkpoint configuration
CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_INTERVAL = 600  # Save every 10 minutes (in seconds)


# ─── CLI Argument Parser ─────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='A2C Multi-Agent Training — Single or Multi-GPU',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--num-gpus', type=int, default=1, choices=[1, 2],
                        help='Number of GPUs to use (1 = single GPU, 2 = dual GPU with MirroredStrategy)')
    parser.add_argument('--num-agents', type=int, default=32,
                        help='Number of parallel agent workers')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Learner batch size')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Resume training from checkpoint (default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                        help='Start fresh training, ignore checkpoints')
    parser.add_argument('--gpu-memory-limit', type=int, default=None,
                        help='Per-GPU memory limit in MB (e.g. 20000 for RTX 3090). '
                             'If not set, memory growth is used instead.')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision (float16) training for faster RTX performance')
    return parser.parse_args()


# ─── GPU Configuration ───────────────────────────────────────────────────────
def configure_gpus(args):
    """
    Detect and configure GPUs. Returns a tf.distribute.Strategy:
      - MirroredStrategy when --num-gpus 2
      - Default (single device) strategy otherwise
    """
    # Set env vars BEFORE any TF ops
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print('[GPU] No GPUs detected — training will run on CPU.')
        return tf.distribute.get_strategy()  # default / CPU strategy

    # Print detected GPUs
    for i, gpu in enumerate(gpus):
        print(f'[GPU] Detected GPU {i}: {gpu.name}')

    # Apply memory configuration to every physical GPU
    try:
        for gpu in gpus:
            if args.gpu_memory_limit:
                # Hard memory cap — useful when sharing a GPU or to leave headroom
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_memory_limit)]
                )
                print(f'[GPU] Set {gpu.name} memory limit to {args.gpu_memory_limit} MB')
            else:
                # Dynamic memory growth — recommended for RTX cards
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f'[GPU] Enabled memory growth for {gpu.name}')
    except RuntimeError as e:
        print(f'[GPU] Configuration error (must be set before TF init): {e}')

    # Mixed precision (optional — RTX Ampere/Ada support FP16 tensor cores)
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print('[GPU] Mixed precision (float16) enabled')

    # Build distribution strategy
    if args.num_gpus == 2:
        if len(gpus) < 2:
            print(f'[GPU] WARNING: Requested 2 GPUs but only {len(gpus)} found. '
                  f'Falling back to single-GPU training.')
            return tf.distribute.get_strategy()
        strategy = tf.distribute.MirroredStrategy()
        print(f'[GPU] MirroredStrategy active — distributing across {strategy.num_replicas_in_sync} GPUs')
        return strategy
    else:
        print('[GPU] Single-GPU training mode')
        return tf.distribute.get_strategy()

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
    step = 0
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
            step += 1
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
                # Use simple reward combination without aggressive normalization
                # The learner will handle normalization with running statistics
                reward_unreal_arr = np.array(reward_unreal, dtype=np.float32)
                reward_real_arr = np.array(reward_real, dtype=np.float32)
                
                # Combine rewards with weighting (unrealized + realized)
                balanced_reward = (0.3 * reward_unreal_arr) + (0.7 * reward_real_arr)
                
                # Light clipping to prevent extreme outliers
                balanced_reward = np.clip(balanced_reward, -500.0, 500.0)
                
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
            
            if agent_id == 0 and step % 1000 == 0:
                print(f"[{agent_id}]: step={step}, {env.chunk_time_step}, {datetime.fromtimestamp(env.chunk_data[-1][-1])}, {env.balance}")
        
        # Save final metrics when loop completes (only save once at the end to avoid I/O overhead)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_metrics_file = f'{path}/final_metrics_{timestamp_str}.json'
        with open(final_metrics_file, 'w') as f:
            json.dump(performance_metrics, f, indent=2)
        logger.info(f"Saved final metrics to {final_metrics_file}")
        print(f"[Agent {agent_id}] Saved {len(performance_metrics['balance_history'])} steps to {final_metrics_file}")

def learner(global_memory_, lock_, checkpoint_mgr, num_gpus=1, gpu_memory_limit=None,
            mixed_precision=False, resume_training=True):
    """
    Central learner process — trains the global model on batched experiences.
    When num_gpus=2, uses MirroredStrategy to distribute across both GPUs.
    """
    # Configure GPUs inside this spawned process (TF config must happen per-process)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if gpu_memory_limit:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_memory_limit)]
                    )
                else:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[Learner] GPU configuration error: {e}")

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print('[Learner] Mixed precision (float16) enabled')

    # Build distribution strategy
    strategy = None
    if num_gpus == 2 and len(gpus) >= 2:
        strategy = tf.distribute.MirroredStrategy()
        print(f'[Learner] MirroredStrategy active — {strategy.num_replicas_in_sync} GPUs')
    else:
        if num_gpus == 2 and len(gpus) < 2:
            print(f'[Learner] WARNING: Requested 2 GPUs but only {len(gpus)} found. Using single GPU.')
        print(f'[Learner] Single-GPU training mode (detected {len(gpus)} GPU(s))')

    # Create the global agent (with multi-GPU strategy if applicable)
    agent = Agent(alpha_actor=ALPHA_ACTOR, alpha_critic=ALPHA_CRITIC, gamma=GAMMA,
                  action_size=ACTION_SIZE, strategy=strategy)
    batch_size = 64
    
    # Try to load checkpoint
    learner_state = checkpoint_mgr.load_learner_state()
    if learner_state and resume_training:
        update_count = learner_state['update_count']
        best_avg_reward = learner_state['best_avg_reward']
        reward_history = learner_state['reward_history']
        try:
            agent.load_model()  # Restore actual network weights (critical for resume!)
            print(f"[Learner] Resuming from checkpoint at update {update_count}, best reward: {best_avg_reward:.4f}")
        except Exception as e:
            print(f"[Learner] Could not load model weights ({e}), starting fresh weights")
            update_count = 0
            best_avg_reward = float('-inf')
            reward_history = []
    else:
        update_count = 0
        best_avg_reward = float('-inf')
        reward_history = []
        print("[Learner] Starting fresh training")
    
    checkpoint_interval = 1000  # Save checkpoint every 1000 updates
    last_checkpoint_time = datetime.now()
    
    # Running statistics for reward normalization
    reward_mean = 0.0
    reward_var = 1.0
    reward_count = 0
    alpha_stats = 0.99  # Exponential moving average coefficient
    
    while True:
        # print(len(global_memory_), len(global_memory_) >= 64)
        if len(global_memory_) >= batch_size:  # Wait until we have enough experiences
            # print('mem is full')
            with lock_:
                batch = global_memory_[:batch_size]
                del global_memory_[:batch_size]

            # Extract rewards and update running statistics
            states, actions, rewards, next_states = zip(*batch)
            rewards_array = np.array(rewards, dtype=np.float32)
            
            # Update running mean and variance
            batch_mean = np.mean(rewards_array)
            batch_var = np.var(rewards_array)
            
            if reward_count == 0:
                reward_mean = batch_mean
                reward_var = batch_var
            else:
                reward_mean = alpha_stats * reward_mean + (1 - alpha_stats) * batch_mean
                reward_var = alpha_stats * reward_var + (1 - alpha_stats) * batch_var
            
            reward_count += 1
            
            # Normalize rewards using running statistics
            reward_std = np.sqrt(reward_var) + 1e-8
            normalized_rewards = [(r - reward_mean) / reward_std for r in rewards]
            normalized_rewards = [np.clip(r, -10.0, 10.0) for r in normalized_rewards]
            
            # Reconstruct batch with normalized rewards
            normalized_batch = list(zip(states, actions, normalized_rewards, next_states))
            
            # Track average reward before normalization
            avg_reward = np.mean([np.mean(r) for r in rewards])
            reward_history.append(avg_reward)
            
            # Perform batch update
            grad_norm = agent.batch_learn(normalized_batch)
            agent.save_sync_model()
            
            update_count += 1
            
            # Periodic checkpoint saving (every 1000 updates)
            if update_count % checkpoint_interval == 0:
                recent_avg = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
                
                print(f"[Learner] Update {update_count} | Avg Reward: {avg_reward:.4f} | "
                      f"Rolling Avg (100): {recent_avg:.4f} | Best: {best_avg_reward:.4f} | "
                      f"Grad Norm: {grad_norm:.4f} | Reward Mean: {reward_mean:.2f} | Std: {reward_std:.2f}")
                
                # Save checkpoint
                try:
                    agent.save_model()
                    if recent_avg > best_avg_reward:
                        best_avg_reward = recent_avg
                        print(f"[Learner] New best model! Avg reward: {best_avg_reward:.4f}")
                        agent.save_best_model()
                except Exception as e:
                    print(f"[Learner] WARNING: Failed to save model weights: {e}")
            
            # Time-based checkpoint saving (every 10 minutes)
            current_time = datetime.now()
            if (current_time - last_checkpoint_time).total_seconds() >= CHECKPOINT_INTERVAL:
                try:
                    checkpoint_mgr.save_learner_state(update_count, best_avg_reward, reward_history)
                    agent.save_model()
                    print(f"[Learner] Saved checkpoint at update {update_count}")
                except Exception as e:
                    print(f"[Learner] WARNING: Failed to save checkpoint: {e}")
                last_checkpoint_time = current_time



if __name__ == '__main__':
    # ─── Parse CLI Arguments ─────────────────────────────────────────────
    args = parse_args()

    # Override globals from CLI
    NUM_AGENTS = args.num_agents
    BATCH_SIZE = args.batch_size
    EPOCHES = args.epochs
    RESUME_TRAINING = args.resume

    # ─── Configure GPUs (main process — for startup diagnostics) ────────
    main_strategy = configure_gpus(args)

    print(f"\n{'='*60}")
    print(f"  A2C Training Configuration")
    print(f"  GPUs requested : {args.num_gpus}")
    print(f"  Agents         : {NUM_AGENTS}")
    print(f"  Batch size     : {BATCH_SIZE}")
    print(f"  Resume         : {RESUME_TRAINING}")
    print(f"  Mixed precision: {args.mixed_precision}")
    if args.gpu_memory_limit:
        print(f"  GPU mem limit  : {args.gpu_memory_limit} MB")
    print(f"{'='*60}\n")

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR)
    
    # Check for existing checkpoint
    resume_info = checkpoint_mgr.get_resume_info()
    print(resume_info['message'])
    
    # Save initial training state
    checkpoint_mgr.save_training_state(
        NUM_AGENTS, START_TRAINING, END_TRAINING, INSTRUMENT, INDICATORS
    )
    
    multiprocessing.set_start_method('spawn')
    manager = multiprocessing.Manager()
    queue = multiprocessing.Queue()
    global_memory = manager.list()
    lock = manager.Lock()
    
    # Graceful shutdown handler
    def signal_handler(sig, frame):
        print('\n\n[Main] Received interrupt signal. Saving checkpoint and shutting down gracefully...')
        import time
        time.sleep(2)
        print('[Main] Checkpoint saved. Exiting.')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    # ─── Launch agent workers ────────────────────────────────────────────
    processes = []
    for i in range(NUM_AGENTS):
        p = multiprocessing.Process(target=agent_worker, args=(i, global_memory, lock, queue))
        processes.append(p)
        p.start()

    # ─── Launch learner (with GPU strategy config passed through) ────────
    learner_process = multiprocessing.Process(
        target=learner,
        args=(global_memory, lock, checkpoint_mgr),
        kwargs={
            'num_gpus': args.num_gpus,
            'gpu_memory_limit': args.gpu_memory_limit,
            'mixed_precision': args.mixed_precision,
            'resume_training': RESUME_TRAINING,
        }
    )
    learner_process.start()

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
