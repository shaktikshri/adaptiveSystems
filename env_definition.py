import numpy as np


class RandomVariable:
    def __init__(self, highest, intermediate, penalty, lowest):
        self.state = 0
        self.time = 0
        self.timestep = 0.5
        self.cur_state = self.f(self.time) # the function is currently in one dimension, we can raise this dimension
        self.observation_space = np.array([1,2]).reshape(-1)
        self.action_space = np.array(1).reshape(-1)
        self.highest = highest
        self.intermediate = intermediate
        self.lowest = lowest
        self.penalty = penalty
        self.max_time = 2*np.pi
        self.counter = 0

    def get_noise(self):
        # make noise as a function of time, or some unique mapping.
        # TODO : sample this from a gaussian centered at self.time
        return [0.1, 0.5, -0.3, -0.5][int(self.time // 2)]

    def reset(self):
        # TODO : This has to start all over again with randomness
        self.time = 0
        # return a noisy function output
        self.cur_state = self.f(self.time) + self.get_noise()
        return np.array([self.time, self.cur_state])

    def f(self, x):
        return np.sin(x)

    def step(self, action):
        self.counter += 1
        if self.counter >= 200:
            reward = self.lowest
            done = True
        else:
            difference = np.abs(self.f(self.time) - (self.cur_state + action))
            done = False
            if difference <= 0.01:
                reward = self.highest
            elif difference <= 0.1:
                reward = self.intermediate
            elif difference <= 0.7:
                reward = self.penalty
            else:
                reward = self.lowest
            # Thus reward directly depends on how good the approximation was
            self.time = (self.time + self.timestep) % self.max_time
            self.cur_state = self.f(self.time) + self.get_noise()
        return np.array([self.time, self.cur_state]), reward, done, 'info'


# # testing the noise functionality
# import matplotlib.pyplot as plt
# env = RandomVariable(10, 5, -1, -5)
# env.reset()
# array1 = list()
# array2 = list()
# for el in range(100):
#     out, _, _, _ = env.step(0)
#     array1.append(out[0])
#     array2.append(out[1])
# plt.scatter(array1, array2)