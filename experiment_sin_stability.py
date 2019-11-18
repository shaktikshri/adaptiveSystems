import numpy as np
import matplotlib.pyplot as plt
from environment import RandomVariable
from util import *

# In[]:

NUM_STATES = 7
NUM_ACTIONS = 10
MAX_EPISODE_LENGTH = 1000
N_EPOCHS = 50000

env = RandomVariable()
gamma = 0.99
print_epoch = 50

# Define the state matrix, there are 7 possible states,
# 2 states on the right side of origin
# 2 states on the left side of origin
# 1 state around the origin
# Thus state 0 is – metric between [-inf, -10)
# Thus state 1 is – metric between [-10, -5)
# Thus state 2 is – metric between [-5, -3)
# Thus state 3 is – metric between [-3, +3]
# Thus state 4 is – metric between (+3, +5]
# Thus state 5 is – metric between (-5, +10]
# Thus state 6 is – metric between (+10, +inf)
state_matrix = np.zeros((NUM_STATES,1))
state_matrix[2] = 1 # this is the safe state, terminal state
state_matrix[0] = state_matrix[6] = 1 # These are the incident state, which is again a terminal state
# We dont need the state matrix since the state depends purely on the value of the metric
# state matrix is depicted only for your understanding

# There are 4 possible actions
# Entry 0 : In State 1: Add 10 to the metric
# Entry 1 : In State 2: Add 5 to the metrix
# Entry 2 : In State 4: Subtract 5 from the metric
# Entry 3 : In State 5: Subtract 10 from the metric
# Nothing has to be done in states 0,3 and 6 since they are the terminal states
action_matrix = np.linspace(start=-10, stop=10, num=NUM_ACTIONS)

env.set_action_to_value_mapping(action_matrix)

# define the reward matrix as per the states,
# State 0 and 6 are incident -> reward -1
# State 1 and 5 are critical -> reward -0.5
# State 2 and 4 are unsafe -> reward -0.1
reward_matrix = np.array([
    -1, -0.1, -0.05, 10, -0.05, -0.1, -1
])
env.set_reward_matrix(reward_matrix)

# Random policy matrix
policy_matrix = np.random.randint(low=0, high=NUM_ACTIONS, size=(NUM_STATES,))
policy_matrix[0] = policy_matrix[6] = -1 # these are the terminal states
policy_matrix[3] = -1 # these are the terminal states

# State-action matrix or the Q values (init to zeros or to random values)
state_action_matrix = np.random.random_sample((NUM_ACTIONS, NUM_STATES))
running_mean_matrix = np.full((NUM_ACTIONS, NUM_STATES), 1.0e-12)
# one row of all states for each action, thus NUM_STATES columns for each row

# IMPORTANT
# reward_matrix, policy_matrix, state_action_matrix, action_to_value_mapping
# and running_mean_matrix are the only ones that we need
# Since the state depends on the current metric value only, we dont need to store the state_matrix in our env variable

all_episode_lists = list()
# for epoch in range(N_EPOCHS):

def perform_generalized_policy_iteration():
    global print_epoch, gamma, NUM_ACTIONS, NUM_STATES, state_action_matrix, \
        policy_matrix, running_mean_matrix, MAX_EPISODE_LENGTH, env

    epoch = 0
    gamma = 0.9

    while True:
        epoch += 1
        episode_list = list()
        observation = env.reset(exploring_starts=True)
        # observation is the current state and the current value
        # which the agent observes

        done = False
        # max length of each episode is 1000
        for _ in range(MAX_EPISODE_LENGTH):
            action = policy_matrix[observation[0]]

            # Move one step and get a new observation and the reward
            new_state, new_value, reward, done = env.step(action)
            new_observation = [new_state, new_value]

            # append what had you observed, and what action did you take resulting in what reward
            episode_list.append((observation, action, reward))
            observation = new_observation
            if done:
                break

        # For debugging
        all_episode_lists.append(episode_list)

        # This cycle is the implementation of First-Visit MC.
        first_visit_done = np.zeros((NUM_ACTIONS, NUM_STATES))
        counter = 0
        # For each state-action stored in the episode list it checks if
        # it is the first visit and then estimates the return.
        # This is the Evaluation step of the GPI.
        old_state_action_matrix = state_action_matrix.copy()
        for visit in episode_list:
            state = visit[0][0]
            action = int(visit[1])
            if first_visit_done[action, state] == 0:
                return_value = get_return(episode_list[counter:], gamma)
                running_mean_matrix[action, state] += 1
                state_action_matrix[action, state] += return_value
                first_visit_done[action, state] = 1
            counter += 1
        # Policy update (Improvement)

        if has_converged(old_state_action_matrix/running_mean_matrix, state_action_matrix/running_mean_matrix):
            break

        policy_matrix = update_policy(episode_list, policy_matrix, state_action_matrix/running_mean_matrix)

        if epoch % print_epoch == 0:
            print("State-Action matrix after " + str(epoch) + " iterations:")
            print(state_action_matrix / running_mean_matrix)
            print("Policy matrix after " + str(epoch + 1) + " iterations:")
            print(policy_matrix)
            describe_policy_matrix(policy_matrix, env)

    # print("Utility matrix after " + str(N_EPOCHS) + " iterations: ")
    # print(state_action_matrix/running_mean_matrix)
    # print('Current Learnt Policy is ')
    # describe_policy_matrix(policy_matrix)