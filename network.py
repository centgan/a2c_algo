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

        # self.NN = nn.Sequential(
        #     nn.LSTM(input_size, self.dims_1, batch_first=True),
        #     nn.LeakyReLU(negative_slope=0.05),
        #     nn.LSTM(self.dims_1, self.dims_2, batch_first=True),
        #     nn.LeakyReLU(negative_slope=0.05),
        #     nn.Linear(self.dims_2, self.action_size),
        #     nn.Softmax(dim=-1)
        # )
        self.lstm1 = nn.LSTM(input_size, self.dims_1, batch_first=True)
        self.lstm2 = nn.LSTM(self.dims_1, self.dims_2, batch_first=True)
        self.fc = nn.Linear(self.dims_2, self.action_size)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.05)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        lstm_out1, _ = self.lstm1(state)  # Output: (batch_size, seq_len, dims_1)
        lstm_out1 = self.leaky_relu(lstm_out1)

        # Second LSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)  # Output: (batch_size, seq_len, dims_2)
        lstm_out2 = self.leaky_relu(lstm_out2)
        lstm_out2_last = lstm_out2[:, -1, :]

        # Output layer
        action_logits = self.fc(lstm_out2_last)  # Output: (batch_size, seq_len, action_size)
        action_probs = self.softmax(action_logits)
        return action_probs
        # pi = self.NN(state)
        #
        # return pi


class CriticNetwork(nn.Module):
    def __init__(self, input_size, dims_1=64, dims_2=32, name='critic', chkpt_dir='tmp/actor_critic'):
        super(CriticNetwork, self).__init__()
        self.dims_1 = dims_1
        self.dims_2 = dims_2
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_first.weights.h5')

        # self.NN = nn.Sequential(
        #     nn.LSTM(input_size, self.dims_1, batch_first=True),
        #     nn.LeakyReLU(negative_slope=0.05),
        #     nn.LSTM(self.dims_1, self.dims_2, batch_first=True),
        #     nn.LeakyReLU(negative_slope=0.05),
        #     nn.Linear(self.dims_2, 1),
        #     nn.Tanh()
        # )
        self.lstm1 = nn.LSTM(input_size, self.dims_1, batch_first=True)
        self.lstm2 = nn.LSTM(self.dims_1, self.dims_2, batch_first=True)
        self.fc = nn.Linear(self.dims_2, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.05)
        self.tanh = nn.Tanh()


    def forward(self, state):
        lstm_out1, _ = self.lstm1(state)  # Output: (batch_size, seq_len, dims_1)
        lstm_out1 = self.leaky_relu(lstm_out1)

        # Second LSTM layer
        lstm_out2, _ = self.lstm2(lstm_out1)  # Output: (batch_size, seq_len, dims_2)
        lstm_out2 = self.leaky_relu(lstm_out2)

        # Output layer
        action_logits = self.fc(lstm_out2)  # Output: (batch_size, seq_len, action_size)
        action_probs = self.tanh(action_logits)
        # print(action_probs.shape, 'critic')
        return action_probs
        # v = self.NN(state)
        #
        # return v

