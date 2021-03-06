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
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.Sigmoid() # sigmoid is needed to bring the output between 0 to 1, later in forward function we'll
                # transform this to -1 to 1
            )
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
        # transform the output between -1 to 1 for continuous action spaces
        if self.continuous:
            out = 2*out - 1
        return out

    def select_action(self, current_state):
        """
        selects an action as per some decided exploration
        :param current_state: the current state
        :return: the chosen action and the log probility to act as the gradient

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
            # keep sampling new actions until it is within -1 to +1.
            while not (explored_action <= +1 and explored_action >= -1):
                explored_action = m.sample()
            # Note that the log prob should be at the original action, not at the exploration since the gradient used
            # will be the gradient of actor's prediction, not of actor's exploration
            return explored_action, m.log_prob(action)
