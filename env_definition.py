# The Purpose of this experiment is to show that RL can even learn latent independent variables
# of a function which the reward structure aims to approximate.
# The latent variable here is the time t, and the output is a direct consequence of this independent
# variable

import numpy as np


class RandomVariable:
    def __init__(self):
        self.state = 0
        self.time = 0
        self.timestep = 0.1
        self.cur_state = self.f(self.time) # the function is currently in one dimension, we can raise this dimension

    def get_noise(self):
        # TODO: Replace this with a gaussian, np.random.normal(0, 0.1)
        return np.random.choice([-0.141, -0.8, 0.12, 0.5])

    def reset(self):
        self.time = np.random.random()
        # return a noisy function output
        self.cur_state = self.f(self.time) + self.get_noise()
        return self.cur_state

    def f(self, x):
        return np.sin(x)

    def step(self, action):
        difference = np.abs(self.f(self.time) - (self.cur_state + action))
        done = False
        reward = -99
        if difference <= 0.01:
            reward = +1
        elif difference <= 0.1:
            reward = -1
        # a max deviation between -0.1 to +0.1 is tolerated, after that the episode ends
        else:
            done = True
        # Thus reward directly depends on how good the approximation was
        self.time += self.timestep
        self.cur_state = self.f(self.time) + self.get_noise()
        return self.cur_state, reward, done, 'info'
