import torch
from torch.optim import Adam
import torch.distributions as dist
from network import ActorNetwork, CriticNetwork
import logging
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, alpha_actor=1., alpha_critic=1., gamma=0.99, action_size=1, input_size=120):
        self.action_size = action_size
        self.gamma = gamma
        self.action = None
        self.action_space = [i for i in range(self.action_size)]

        self.actor = ActorNetwork(action_size=action_size, input_size=input_size).to(device)
        self.critic = CriticNetwork(input_size=input_size).to(device)

        self.actor.optimizer = Adam(self.actor.parameters(), lr=alpha_actor)
        self.critic.optimizer = Adam(self.critic.parameters(), lr=alpha_critic)

        self.balance = 0

        self.logger = logging.getLogger()
        logging.basicConfig(filename='log.log', level=logging.INFO,
                            format='%(asctime)s  %(levelname)s: %(message)s')

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).to(device)
        probs = self.actor(state)
        probs = probs / probs.sum(dim=1, keepdim=True)

        action_probs = dist.Categorical(probs=probs)
        action = action_probs.sample()

        self.action = action

        return action.detach().cpu().numpy()

    def learn(self, state,  reward, state_):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        state_ = torch.tensor(state_, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)

        self.actor.train()
        self.critic.train()

        state_val = self.critic(state)
        probs = self.actor(state)
        state_val_ = self.critic(state_)

        state_val = state_val.squeeze(-1)
        state_val_ = state_val_.squeeze(-1)
        state_val = state_val[:, -1]  # Get value at the last time step (shape: [256])
        state_val_ = state_val_[:, -1]
        action_probs = dist.Categorical(probs=probs)
        actions = action_probs.sample()

        log_prob = action_probs.log_prob(actions)
        delta = reward + self.gamma * state_val_ - state_val  # Compute the advantage (delta)
        actor_loss = -log_prob * delta  # Actor loss: negative log probability * advantage
        critic_loss = delta ** 2  # Critic loss: squared error of the advantage

        # Zero the gradients for actor and critic before backpropagation
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # Backpropagate for actor loss
        actor_loss.mean().backward(retain_graph=True)
        self.actor.optimizer.step()

        # Backpropagate for critic loss
        critic_loss.mean().backward()
        self.critic.optimizer.step()


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