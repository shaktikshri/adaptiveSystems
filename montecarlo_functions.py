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
# each row is probabilities for next action given the current action,
# row0 is prob. for next actions given current action is UP
# row1 is prob. for next actions given current action is RIGHT
# row2 is prob. for next actions given current action is DOWN
# row3 is prob. for next actions given current action is LEFT
# Thus its basically the prob. of taking the initial actions.
# TODO : Should check if this also forms part of learning in model-free RL
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
# and return the first observation.
# Observation is the current row and column, hence the current state
observation = env.reset()
# Print the current world in the terminal.
env.render()

# In[]:
# Now we can run an episode using a for loop:
# This is just playing an episode,
# there is no learning here since the transition matrix was defined already
for _ in range(1000):
    action = policy_matrix[observation[0], observation[1]]
    # Now the robot should move one step in the world based on the action given.
    # The action can be 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    observation, reward, done = env.step(action)
    # done is True if the current state is a terminal state, i.e. either the charging station
    # or the staircase. (Note that the invalid state cannot be the terminal state)
    print("")
    print("ACTION: " + str(action))
    print("REWARD: " + str(reward))
    print("DONE: " + str(done))
    env.render()
    if done:
        break