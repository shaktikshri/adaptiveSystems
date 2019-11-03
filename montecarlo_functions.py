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

# In[]:
# In this section we are going to use the Monte Carlo for Prediction, thus
# to find the value/utility function. This means that we already have the
# transition function with us, and we only need to estimate the values of the
# states now


def get_return(state_list, gamma):
    """
    :param state_list: a list of tuples (state, reward)
    :param gamma: the discount factor
    :return: the return value for that state_list
    """
    return_value = 0
    counter = 0
    for visit in state_list:
        reward = visit[1]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value


# We are going to use the function get_return in the following loop in order
# to get the returns for each episode and estimate the utilities/value function
utility_matrix = np.zeros((3, 4))
running_mean_matrix = np.full((3,4), 1.0e-10)
gamma = 1
n_epochs = 50000
print_epoch = 1000

for epoch in range(n_epochs):
    episode = list()
    observation = env.reset(exploring_starts=False)
    for _ in range(1000):
        action = policy_matrix[observation[0], observation[1]]
        observation, reward, done = env.step(action=action)
        episode.append((observation, reward))
        if done:
            break
    first_visit_done = np.zeros((3,4))
    counter = 0
    for visit in episode:
        observation = visit[0]
        reward = visit[1]
        row = observation[0]
        column = observation[1]
        if first_visit_done[row, column] == 0:
            return_value = get_return(episode[counter:], gamma)
            running_mean_matrix[row, column] += 1
            utility_matrix[row, column] += return_value
            first_visit_done[row, column] = 1
        counter += 1
    if epoch % print_epoch == 0:
        print('Utility matrix after '+str(epoch)+" iterations : ")
        print(utility_matrix / running_mean_matrix)

print('Utility matrix after ' + str(n_epochs) + " iterations : ")
print(utility_matrix / running_mean_matrix)

# In[]:
# Until now we used the function called the utility function (aka value function,
# state-value function) as a way to estimate the utility (value) of a state. We
# still havent found a way to get the optimal policy
# Thus the next section is MC for control. Its estimating the Q(s,a) function or
# finding out utility of each (action, state) pair. Once we have found Q,
# the action at each state would simply be action(s) = argmax(a) Q(s, a) i.e.
# take the action that maximizes the utility of that state

def get_return(state_list, gamma):
    """
    :param state_list: a list of tuples (observation, state, reward)
    :param gamma: the discount factor
    :return: the return value for that state_list
    """
    return_value = 0
    counter = 0
    for visit in state_list:
        reward = visit[2]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value


def update_policy(episode_list, policy_matrix, state_action_matrix):
    """
    Updates the policy in a greedy way, selecting actions which have the highest
    utility for each state visited in the episode_list
    :param episode_list: the tuples of states visited as (observation, state, reward)
    :param policy_matrix: the policy matrix
    :param state_action_matrix: the Q matrix
    :return:
    """
    for visit in episode_list:
        observation = visit[0]
        column = observation[1] + observation[0] * 4
        if policy_matrix[observation[0], observation[1]] != -1:
            # if its not the terminal state
            policy_matrix[observation[0], observation[1]] = np.argmax(state_action_matrix[:, column])
    return policy_matrix

# Random policy matrix
policy_matrix = np.random.randint(low=0, high=4, size=(3, 4)).astype(np.float32)
policy_matrix[1, 1] = np.NaN  # NaN for the obstacle at (1,1)
policy_matrix[0, 3] = policy_matrix[1,3] = -1  # No action (terminal states)

# State-action matrix or the Q values (init to zeros or to random values)
state_action_matrix = np.random.random_sample((4, 12))
# one row of all states for each action, thus 12 columns for each row
from rl_prc import print_policy
n_epochs = 50000
for epoch in range(n_epochs):
    episode_list = list()
    observation = env.reset(exploring_starts=True)
    is_starting = True
    # length of each episode is 1000
    for _ in range(1000):
        action = policy_matrix[observation[0], observation[1]]
        # If the episode just started then it is
        # necessary to choose a random action (exploring starts)
        # This condition assures to satisfy the exploring starts. T
        if is_starting:
            action = np.random.randint(0, 4)
            is_starting = False
        # Move one step and get a new observation and the reward
        new_observation, reward, done = env.step(action)
        episode_list.append((observation, action, reward))
        observation = new_observation
        if done:
            break
    # This cycle is the implementation of First-Visit MC.
    first_visit_done = np.zeros((4, 12))
    counter = 0
    # For each state-action stored in the episode list it checks if
    # it is the first visit and then estimates the return.
    # This is the Evaluation step of the GPI.
    for visit in episode_list:
        observation = visit[0]
        action = visit[1]
        column = observation[1] + observation[0] * 4
        row = action
        if first_visit_done[row, column] == 0:
            return_value = get_return(episode_list[counter:], gamma)
            running_mean_matrix[row, column] += 1
            state_action_matrix[row, column] += return_value
            first_visit_done[row, column] = 1
        counter += 1
    # Policy update (Improvement)
    policy_matrix = update_policy(episode_list, policy_matrix, state_action_matrix/running_mean_matrix)

    if epoch % print_epoch == 0:
        print("State-Action matrix after " + str(epoch) + " iterations:")
        print(state_action_matrix / running_mean_matrix)
        print("Policy matrix after " + str(epoch + 1) + " iterations:")
        print(policy_matrix)
        print_policy(policy_matrix, (3,4))

print("Utility matrix after " + str(n_epochs) + " iterations:")
print(state_action_matrix / running_mean_matrix)

