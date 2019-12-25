import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=12):
        super(Actor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.output_layer(out)
        return out

env = gym.make('CartPole-v1')
actor = Actor(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
critic = Actor(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
