# In[]:

import numpy as np
from gridworld import GridWorld

# In[]:

env = GridWorld(3, 4)
state_matrix = np.zeros((3,4))
# this is the charging station
state_matrix[0, 3] = 1
# this is the staircase
state_matrix[1, 3] = 1
# this is the invalid state
state_matrix[1, 1] = -1

reward = np.full((3,4), -0.04)
# this is the staircase
reward[1, 3] = -1
# this is the charging station
reward[0, 3] = 1

# For each one of the four actions there is a probability, 0th index is UP, 1: RIGHT, 2:DOWN, 3:LEFT
# TODO : Check the significance of this
transition_matrix = np.array([
    [0.8, 0.1, 0.0, 0.1],
    [0.1, 0.8, 0.1, 0.0],
    [0.0, 0.1, 0.8, 0.1],
    [0.1, 0.0, 0.1, 0.8]
])

# Define the policy matrix
# 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, NaN=Obstacle, -1=NoAction
policy_matrix = np.array([
    [1, 1, 1, -1],
    [0,np.NaN, 0, -1],
    [0, 3, 3, 3]
])

# Set the matrices now
env.setStateMatrix(state_matrix)
env.setRewardMatrix(reward)
env.setTransitionMatrix(transition_matrix)

# Set the position of the robot in the bottom left corner.
# and return the first observation
observation = env.reset()
# Print the current world in the terminal.
env.render()

# In[]: