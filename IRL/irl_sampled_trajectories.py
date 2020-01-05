# this is learning reward function from a sampled trajectory


import numpy as np

# constructing a 5*5 gridworld as put by Andrew and Russel in Inverse Reinforcement Learning
# state is continuous [0,1]*[0,1]
cur_state = np.random.rand(2)
class Agent:
    def __init__(self):
        self.cur_state = np.random.rand(2)
    def step(self, action):
        assert action in [0, 1, 2, 3]
        # UP is 0, RIGHT 1, DOWN 2, LEFT 3
        if action == 0:
            step_size = np.array([0, 0.2])
        elif action == 1:
            step_size = np.array([0.2, 0])
        elif action == 2:
            step_size = np.array([0, -0.2])
        else:
            step_size = np.array([-0.2, 0])
        # whenever an action is performed, the guy moves in that direction and then a uniform noise from [-0.1, 0.1]
        # is added in each of the coordinates
        self.cur_state = self.cur_state + step_size + \
                         np.array([(2*np.random.rand() - 1)/10, (2*np.random.rand() - 1)/10])
        # Truncate the state to be in [0,1]*[0,1]
        self.cur_state[self.cur_state > 1] = 1
        self.cur_state[self.cur_state < 0] = 0
        return self.cur_state

"""
Remember whenever you're in lack of domain for basis functions, you can always take the domain to be the same
as the domain of the state space. And as always, Gaussian Mixtures are the best choice for basis functions
"""
# generate evenly spaced 15*15 2d gaussian over the state space
cov = [[0.1, 0],[0, 0.1]]
std = 0.1
mean = np.arange(0, 1, 1/15)
from scipy.stats import multivariate_normal

basis = np.array([])
for i in range(15):
    for j in range(15):
        basis = np.append(basis, multivariate_normal(mean=[mean[i], mean[j]], cov=cov))
basis = basis.reshape(15, 15)
