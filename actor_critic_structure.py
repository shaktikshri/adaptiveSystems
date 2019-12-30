import torch.nn as nn
import torch.optim as optim
import gym
import torch
from dqn import ReplayBuffer
from torch.distributions import Categorical, Normal
from torch.nn.functional import mse_loss
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# In[]:

# TODO :
#  2. Fixing target


class Critic(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=12):
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return out


class QCritic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=12):
        super(QCritic, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return out


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=12, continuous=False):
        super(Actor, self).__init__()
        self.continuous = continuous
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        if continuous:
            # if its continuous action space then done use a softmax at the last layer
            self.output_layer = nn.Linear(hidden_size, output_size)
        else:
            # else use a softmax
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                # TODO : Try out log here if any numerical instability occurs
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return out

    def select_action(self, current_state):
        """
        selects an action as per some decided exploration
        :param current_state: the current state
        :return:
        1. if action space is discrete -> the chosen action and the log probility of that chosen action
        2. if action space is continuous -> the predicted action, the explored action and the
        log probability of the predicted action to act as the gradient

        """
        if not self.continuous:
            # if its not continuous action space then use epsilon greedy selection
            probs = self(current_state) # probs is the probability of each of the discrete actions possible
            # No gaussian exploration can be performed since the actions are discrete and not continuous
            # gaussian would make sense and feasibility only when actions are continuous
            m = Categorical(probs)
            action = m.sample()
            return action, m.log_prob(action)
        else:
            # use gaussian or other form of exploration in continuous action space
            action = self(current_state) # action is the action predicted for this current_state
            # now time to explore, so sample from a gaussian distribution centered at action
            # TODO : This scale can be controlled, its the variance around the mean action
            m = Normal(loc=action, scale=torch.Tensor([0.1]))
            explored_action = m.sample()
            return action, explored_action, m.log_prob(action)
