# This file is MC for control. Its estimating the Q(s,a) function or
# finding out utility of each (action, state) pair. Once we have found Q,
# the action at each state would simply be action(s) = argmax(a) Q(s, a) i.e.
# take the action that maximizes the utility of that state
import numpy as np
from gridworld import GridWorld


def get_return(state_list, gamma):
    """
    :param state_list: a list of tuples (observation, action, reward)
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


def print_policy(p, shape):
    """
    Printing utility
    Print the policy actions using symbols:
    ^, v, <, > up, down, left, right
    * terminal states
    # obstacles
    :param p: the policy matrix
    :param shape: the shape of the matrix
    :return: None
    """
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):
            if(p[row][col] == -1): policy_string += " *  "
            elif(p[row][col] == 0): policy_string += " ^  "
            elif(p[row][col] == 1): policy_string += " >  "
            elif(p[row][col] == 2): policy_string += " v  "
            elif(p[row][col] == 3): policy_string += " <  "
            elif(np.isnan(p[row][col])): policy_string += " #  "
        policy_string += '\n'
    print(policy_string)


def update_policy(episode_list, policy_matrix, state_action_matrix):
    """
    Updates the policy in a greedy way, selecting actions which have the highest
    utility for each state visited in the episode_list
    :param episode_list: the tuples of states visited as (observation, action, reward)
    :param policy_matrix: the policy matrix
    :param state_action_matrix: the Q matrix
    :return:
    """
    for visit in episode_list:
        observation = visit[0]
        column = observation[0] * 4 + observation[1]
        if policy_matrix[observation[0], observation[1]] != -1:
            # if its not the terminal state
            policy_matrix[observation[0], observation[1]] = np.argmax(state_action_matrix[:, column])
    return policy_matrix


env = GridWorld(3, 4)
gamma = 0.99
print_epoch = 10000

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
env.setStateMatrix(state_matrix)
env.setRewardMatrix(reward)

# Random policy matrix
policy_matrix = np.random.randint(low=0, high=4, size=(3, 4)).astype(np.float32)
policy_matrix[1, 1] = np.NaN  # NaN for the obstacle at (1,1)
policy_matrix[0, 3] = policy_matrix[1,3] = -1  # No action (terminal states)

# State-action matrix or the Q values (init to zeros or to random values)
Q = np.random.random_sample((4, 12))
running_mean_matrix = np.full((4, 12), 1.0e-12)
# one row of all states for each action, thus 12 columns for each row
n_epochs = 500000

for epoch in range(n_epochs):
    episode_list = list()
    observation = env.reset(exploring_starts=False)
    # observation is the [row,col] of the current position
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
        # append what had you observed, and what action did you take resulting in what reward
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
        column = observation[0] * 4 + observation[1]
        row = int(action)
        if first_visit_done[row, column] == 0:
            return_value = get_return(episode_list[counter:], gamma)
            running_mean_matrix[row, column] += 1
            Q[row, column] += return_value
            first_visit_done[row, column] = 1
        counter += 1
    # Policy update (Improvement)
    policy_matrix = update_policy(episode_list, policy_matrix, Q / running_mean_matrix)

    if epoch % print_epoch == 0:
        print("State-Action matrix after " + str(epoch) + " iterations:")
        print(Q / running_mean_matrix)
        print("Policy matrix after " + str(epoch + 1) + " iterations:")
        print(policy_matrix)
        print_policy(policy_matrix, (3,4))

print("Utility matrix after " + str(n_epochs) + " iterations: ")
print(Q / running_mean_matrix)
