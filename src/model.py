import os

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from src.network import ActorNetwork, CriticNetwork
import logging
import numpy as np

class Agent:
    def __init__(self, alpha_actor=1., alpha_critic=1., gamma=0.99, action_size=1):
        self.action_size = action_size
        self.gamma = gamma
        self.action = None
        self.action_space = [i for i in range(self.action_size)]

        self.actor = ActorNetwork(action_size=action_size)
        self.critic = CriticNetwork()

        self.actor.compile(optimizer=Adam(learning_rate=alpha_actor, clipnorm=1.0))
        self.critic.compile(optimizer=Adam(learning_rate=alpha_critic, clipnorm=1.0))

        self.balance = 0

        self.logger = logging.getLogger()
        logging.basicConfig(filename='../log.log', level=logging.INFO,
                            format='%(asctime)s  %(levelname)s: %(message)s')

        # make the folder for sync weights first and then delete any files that were used in the previous run
        act = self.actor.sync_dir + '/actor.npy'
        crit = self.critic.sync_dir + '/critic.npy'
        os.makedirs(self.actor.sync_dir, exist_ok=True)
        try:
            os.remove(act)
            os.remove(crit)
            # print('deleted actor and critic files')
        except FileNotFoundError:
            pass

    def choose_action(self, observation):
        state = tf.convert_to_tensor(observation)
        probs = self.actor.call(state)
        # probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)

        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        # flipped_action = 2 - action
        self.action = action

        return self.action.numpy()

    def learn(self, state, reward, state_):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            state_val = self.critic.call(state)
            probs = self.actor.call(state)
            state_val_ = self.critic.call(state_)

            state_val = tf.squeeze(state_val)
            state_val_ = tf.squeeze(state_val_)

            action_probs = tfp.distributions.Categorical(probs=probs + 1e-8)
            # action = action_probs.sample()
            # entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * state_val_ - state_val
            delta = (delta - tf.reduce_mean(delta)) / tf.math.reduce_std(delta + 1e-8)

            actor_loss = -log_prob * delta #- 0.05 * entropy
            critic_loss = delta ** 2
            # total_loss = actor_loss + critic_loss

        # gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        # max_grad = max([tf.reduce_max(tf.abs(g)).numpy() for g in gradients if g is not None])
        # print(f"Max Gradient: {max_grad}")
        #
        # min_grad = min([tf.reduce_min(tf.abs(g)).numpy() for g in gradients if g is not None])
        # print(f"Min Gradient: {min_grad}")

        gradient_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            gradient_actor, self.actor.trainable_variables
        ))
        gradient_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            gradient_critic, self.critic.trainable_variables
        ))


    def batch_learn(self, memory_batch):
        state, action, reward, state_ = zip(*memory_batch)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int32)
        with tf.GradientTape(persistent=True) as tape:
            # need to squeeze because lstm only accepts ndim=3 but because of how its set up it is currently a ndim=4 so
            # need to squeeze it
            # print(state.shape)
            state = tf.squeeze(state)
            state_ = tf.squeeze(state_)
            # print(state.shape)

            state_val = self.critic.call(state)
            probs = self.actor.call(state)
            state_val_ = self.critic.call(state_)

            state_val = tf.squeeze(state_val)
            state_val_ = tf.squeeze(state_val_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)

            # is the advantage function
            delta = reward + self.gamma * state_val_ - state_val
            delta = (delta - tf.reduce_mean(delta)) / tf.math.reduce_std(delta + 1e-8)

            actor_loss = -log_prob * delta  # - 0.05 * entropy
            critic_loss = delta ** 2

        gradient_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            gradient_actor, self.actor.trainable_variables
        ))
        gradient_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            gradient_critic, self.critic.trainable_variables
        ))

    def save_sync_model(self):
        # if not self.actor.built:
        #     self.actor.build((None, 60, 8))
        # if not self.critic.built:
        #     self.critic.build((None, 60, 8))
        self.logger.info('... saving temp model for sync ...')
        actor_weights = self.actor.get_weights()
        np.save(self.actor.sync_dir + "/actor.npy", np.array(actor_weights, dtype=object), allow_pickle=True)

        # Save critic
        critic_weights = self.critic.get_weights()
        np.save(self.critic.sync_dir + "/critic.npy", np.array(critic_weights, dtype=object), allow_pickle=True)
        # self.actor.save_weights(self.actor.sync_checkpoint_dir)
        # self.critic.save_weights(self.critic.sync_checkpoint_dir)

    def load_sync_model(self, indicators):
        ind_count = sum(indicators)
        # if not self.actor.built:
        #     self.actor.build((None, 60, 8))
        # if not self.critic.built:
        #     self.critic.build((None, 60, 8))
        self.logger.info('... loading temp model for sync ...')

        actor_weights = np.load(self.actor.sync_dir + "/actor.npy", allow_pickle=True)
        self.actor.set_weights(actor_weights.tolist())

        dummy_input = tf.zeros((1, 60, 5+ind_count))
        self.critic.call(dummy_input)
        # Load critic
        critic_weights = np.load(self.critic.sync_dir + "/critic.npy", allow_pickle=True)
        self.critic.set_weights(critic_weights.tolist())

        # self.actor.load_weights(self.actor.sync_checkpoint_dir)
        # self.critic.load_weights(self.critic.sync_checkpoint_dir)

    def save_model(self):
        # print('... saving model ...')
        self.logger.info('... saving model ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
    
    def save_best_model(self):
        """Save the best model to a separate directory"""
        self.logger.info('... saving best model ...')
        best_dir = 'tmp/best_model'
        os.makedirs(best_dir, exist_ok=True)
        self.actor.save_weights(os.path.join(best_dir, 'actor_best.weights.h5'))
        self.critic.save_weights(os.path.join(best_dir, 'critic_best.weights.h5'))

    def load_model(self):
        # print('... loading model ...')
        self.logger.info('... loading model ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)