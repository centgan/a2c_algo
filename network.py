import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU
from tensorflow.keras import Sequential


class ActorCriticNetwork(keras.Model):
    def __init__(self, action_size, dims_1=64, dims_2=32, name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.dims_1 = dims_1
        self.dims_2 = dims_2
        self.action_size = action_size
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_first.weight.h5')

        self.layer1 = Dense(dims_1, activation='relu')
        self.layer2 = Dense(dims_2, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(self.action_size, activation='softmax')

    def call(self, state):
        value = self.layer1(state)
        value = self.layer2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi


class ActorNetwork(keras.Model):
    def __init__(self, action_size, dims_1=64, dims_2=32, name='actor', chkpt_dir='tmp/actor_critic'):
        super(ActorNetwork, self).__init__()
        self.dims_1 = dims_1
        self.dims_2 = dims_2
        self.action_size = action_size
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_first.weights.h5')

        self.NN = Sequential([
            LSTM(self.dims_1, return_sequences=True),
            LeakyReLU(negative_slope=0.05),
            LSTM(self.dims_2),
            LeakyReLU(negative_slope=0.05),
            Dense(self.action_size, activation='softmax'),
        ])
        # self.layer1 = LSTM(dims_1)
        # self.layer2 = LeakyReLU(alpha=0.05)
        # self.layer3 = LSTM(dims_2)
        # self.layer4 = LeakyReLU(alpha=0.05)
        # self.pi = Dense(self.action_size, activation='softmax')

    def call(self, state):
        # value = self.layer1(state)
        # value = self.layer2(value)

        # v = self.v(value)
        # pi = self.pi(value)

        pi = self.NN(state)
        return pi


class CriticNetwork(keras.Model):
    def __init__(self, dims_1=64, dims_2=32, name='critic', chkpt_dir='tmp/actor_critic'):
        super(CriticNetwork, self).__init__()
        self.dims_1 = dims_1
        self.dims_2 = dims_2
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_first.weights.h5')

        # self.layer1 = Dense(dims_1, activation='relu')
        # self.layer2 = Dense(dims_2, activation='relu')
        # self.v = Dense(1, activation=None)
        # self.pi = Dense(self.action_size, activation='softmax')
        self.NN = Sequential([
            LSTM(self.dims_1, return_sequences=True),
            LeakyReLU(negative_slope=0.05),
            LSTM(self.dims_2),
            LeakyReLU(negative_slope=0.05),
            Dense(1, activation='tanh'),
        ])

    def call(self, state):
        # value = self.layer1(state)
        # value = self.layer2(value)
        #
        # v = self.v(value)
        # pi = self.pi(value)

        v = self.NN(state)

        return v

