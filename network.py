import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LSTM, LeakyReLU, BatchNormalization
from tensorflow.keras import Sequential

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
            LSTM(self.dims_2,return_sequences=False),
            LeakyReLU(negative_slope=0.05),
            Dense(self.action_size, activation='softmax'),
        ])

    def call(self, state):
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

        self.NN = Sequential([
            LSTM(self.dims_1, return_sequences=True),
            # BatchNormalization(),
            LeakyReLU(negative_slope=0.05),
            LSTM(self.dims_2, return_sequences=False),
            # BatchNormalization(),
            LeakyReLU(negative_slope=0.05),
            Dense(1, activation='linear'),
        ])

    def call(self, state):
        v = self.NN(state)

        return v

