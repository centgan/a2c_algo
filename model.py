import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from network import ActorCriticNetwork, ActorNetwork, CriticNetwork
import logging


class Agent:
    def __init__(self, alpha=1, gamma=0.99, action_size=1):
        self.action_size = action_size
        self.gamma = gamma
        self.action = None
        self.action_space = [i for i in range(self.action_size)]

        self.actor_critic = ActorCriticNetwork(action_size=action_size)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

        self.logger = logging.getLogger()
        logging.basicConfig(filename='log.log', level=logging.INFO,
                            format='%(asctime)s  %(levelname)s: %(message)s')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        _, probs = self.actor_critic.call(state)
        print(probs)
        action_probs = tfp.distributions.Categorical(probs=probs)
        print(action_probs)
        action = action_probs.sample()

        self.action = action
        print(action)

        return action.numpy()[0]

    def learn(self, state,  reward, state_):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            state_val, probs = self.actor_critic.call(state)
            state_val_, _ = self.actor_critic.call(state_)
            state_val = tf.squeeze(state_val)
            state_val_ = tf.squeeze(state_val_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(state_val)

            delta = reward + self.gamma * state_val_ - state_val
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables
        ))

    def save_model(self):
        # print('... saving model ...')
        self.logger.info('... saving model ...')
        self.actor_critic.model.save_weights(self.actor_critic.checkpoint_file)

    def load_model(self):
        # print('... loading model ...')
        self.logger.info('... loading model ...')
        self.actor_critic.model.load_weights(self.actor_critic.checkpoint_file)


class AgentSep:
    def __init__(self, alpha_actor=1, alpha_critic=1, gamma=0.99, action_size=1):
        self.action_size = action_size
        self.gamma = gamma
        self.action = None
        self.action_space = [i for i in range(self.action_size)]

        self.actor = ActorNetwork(action_size=action_size)
        self.critic = CriticNetwork()

        self.actor.compile(optimizer=Adam(learning_rate=alpha_actor))
        self.critic.compile(optimizer=Adam(learning_rate=alpha_critic))

        self.logger = logging.getLogger()
        logging.basicConfig(filename='log.log', level=logging.INFO,
                            format='%(asctime)s  %(levelname)s: %(message)s')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor.call(state)

        # print(probs)
        action_probs = tfp.distributions.Categorical(probs=probs)
        # print(action_probs)
        action = action_probs.sample()

        self.action = action
        # print(action)

        return action.numpy()[0]

    def learn(self, state,  reward, state_):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            # state_val, probs = self.actor_critic.call(state)
            state_val = self.critic.call(state)
            probs = self.actor.call(state)
            state_val_ = self.critic.call(state_)
            # state_val_, _ = self.actor_critic.call(state_)
            state_val = tf.squeeze(state_val)
            state_val_ = tf.squeeze(state_val_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(state_val)

            delta = reward + self.gamma * state_val_ - state_val
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            # total_loss = actor_loss + critic_loss

        gradient_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            gradient_actor, self.actor.trainable_variables
        ))
        gradient_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            gradient_critic, self.critic.trainable_variables
        ))

    def save_model(self):
        # print('... saving model ...')
        self.logger.info('... saving model ...')
        self.actor.NN.save_weights(self.actor.checkpoint_file)
        self.critic.NN.save_weights(self.critic.checkpoint_file)

    def load_model(self):
        # print('... loading model ...')
        self.logger.info('... loading model ...')
        self.actor.NN.load_weights(self.actor.checkpoint_file)
        self.critic.NN.load_weights(self.critic.checkpoint_file)