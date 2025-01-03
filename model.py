import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from network import ActorNetwork, CriticNetwork
import logging


class Agent:
    def __init__(self, alpha_actor=1, alpha_critic=1, gamma=0.99, action_size=1):
        self.action_size = action_size
        self.gamma = gamma
        self.action = None
        self.action_space = [i for i in range(self.action_size)]

        self.actor = ActorNetwork(action_size=action_size)
        self.critic = CriticNetwork()

        self.actor.compile(optimizer=Adam(learning_rate=alpha_actor))
        self.critic.compile(optimizer=Adam(learning_rate=alpha_critic))

        self.balance = 0

        self.logger = logging.getLogger()
        logging.basicConfig(filename='log.log', level=logging.INFO,
                            format='%(asctime)s  %(levelname)s: %(message)s')

    def update_balance(self, balance):
        self.balance = balance

    def choose_action(self, observation):
        state = tf.convert_to_tensor(observation)
        # start = time.time()
        probs = self.actor.call(state)

        # actor_time = time.time()
        # print("---to actor call----: ", actor_time - start)
        action_probs = tfp.distributions.Categorical(probs=probs)
        # probs_numpy = probs.numpy()[0]
        # cumulative_probs = np.cumsum(probs_numpy)
        # # Generate a random number
        # random_value = np.random.rand()
        # # Find the category based on the random value
        # for i, cumulative_prob in enumerate(cumulative_probs):
        #     if random_value < cumulative_prob:
        #         action = i
        #         print(i)
        #         break
        # print("---to distribution----: ", time.time() - actor_time)
        # print(action_probs)
        action = action_probs.sample()

        self.action = action
        # print(action)

        return action.numpy()[0]

    def learn(self, state,  reward, state_):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state_ = tf.convert_to_tensor(state_, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            # state_val, probs = self.actor_critic.call(state)
            state_val = self.critic.call(state)
            probs = self.actor.call(state)
            state_val_ = self.critic.call(state_)
            # state_val_, _ = self.actor_critic.call(state_)
            state_val = tf.squeeze(state_val)
            state_val_ = tf.squeeze(state_val_)
            # print(state_val)
            # print(state_val[0])
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(state_val)
            # print('log probs: ' + str(log_prob))

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