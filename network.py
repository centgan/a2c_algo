import os
from torch import nn


class ActorNetwork(nn.Module):
    def __init__(self, action_size, input_size, dims_1=64, dims_2=32, name='actor', chkpt_dir='tmp/actor_critic'):
        super(ActorNetwork, self).__init__()
        self.dims_1 = dims_1
        self.dims_2 = dims_2
        self.action_size = action_size
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_first.weights.h5')

        self.NN = nn.Sequential(
            nn.LSTM(input_size, self.dims_1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.LSTM(self.dims_1, self.dims_2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(self.dims_2, self.action_size),
            nn.Softmax(dim=self.action_size)
        )

    def forward(self, state):
        pi = self.NN(state)

        return pi


class CriticNetwork(nn.Module):
    def __init__(self, input_size, dims_1=64, dims_2=32, name='critic', chkpt_dir='tmp/actor_critic'):
        super(CriticNetwork, self).__init__()
        self.dims_1 = dims_1
        self.dims_2 = dims_2
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_first.weights.h5')

        self.NN = nn.Sequential(
            nn.LSTM(input_size, self.dims_1, batch_first=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.LSTM(self.dims_1, self.dims_2, batch_first=True),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(self.dims_2, 1),
            nn.Tanh()
        )

    def call(self, state):
        v = self.NN(state)

        return v

