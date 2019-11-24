import numpy as np
import matplotlib.pyplot as plt
from env_definition import RandomVariable
from util import *

# In[]:

NUM_STATES = 5
NUM_ACTIONS = 10
MAX_EPISODE_LENGTH = 1000
N_EPOCHS = 50000

env = RandomVariable()

# Define the state matrix, there are 5 possible states,
# 2 states on the right side of origin
# 2 states on the left side of origin
# 1 state around the origin
# Thus state 0 is – metric between normal_value - (-inf, -0.1)
# Thus state 1 is – metric between normal_value - [-0.1, -0.01)
# Thus state 2 is – metric between [normal_value - 0.01] to [normal_value + 0.01]
# Thus state 3 is – metric between normal_value + (0.01, +0.1]
# Thus state 4 is – metric between normal_value + (+0.1, +inf)
state_matrix = np.zeros((NUM_STATES, 1))
state_matrix[0] = state_matrix[-1] = 1 # These are the incident state, which is again a terminal state
# We dont need the state matrix since the state depends purely on the value of the metric
# state matrix is depicted only for your understanding

# There are 10 possible actions
# Each of them given a value to be added into the current value of the function which
# it aims to stabilize. These values are from uniform log scale between -0.5 to +0.5
action_matrix = np.linspace(start=-1, stop=1, num=NUM_ACTIONS)
env.set_action_to_value_mapping(action_matrix)

# We dont need to define the transition matrix as this is a model free Q learning via TD(0) updates. LOL why didnt
# I understand this earlier?
# define the reward matrix as per the states,
# State 0 and 6 are incident -> reward -1
# State 1 and 5 are critical -> reward -0.5
# State 2 and 4 are unsafe -> reward -0.1
reward_matrix = np.array([
    -1, -0.04, +1, -0.04, -1
])
env.set_reward_matrix(reward_matrix)

# We dont need the policy matrix as of now, we'll do a Q learning first
# Random policy matrix
# policy_matrix = np.random.randint(low=0, high=NUM_ACTIONS, size=(NUM_STATES,))
# policy_matrix[0] = policy_matrix[-1] = -1 # these are the terminal states

# State-action matrix or the Q values (init to zeros or to random values)
Q = np.random.random_sample((NUM_STATES, NUM_ACTIONS))
Q_new = np.random.random_sample((NUM_STATES, NUM_ACTIONS))
all_episode_lists = list()

# starting with Q learning now.
timestep = 0.1
epsilon = 0.1
alpha = 0.1
gamma = 0.9
print_episode = 50
difference = 10
very_small = 0.001
TRAIN_EPISODES = 100

for episode in range(TRAIN_EPISODES):
    done = False
    time = 0
    state, function_value = env.reset(exploring_starts=True)
    this_episode = list()
    while not done:
        # draw actions as per epsilon greedy
        choice = np.random.choice(2, p=[epsilon, 1-epsilon])
        if choice == 0:
            # take random action
            action = np.random.choice(action_matrix)
        else:
            action = np.argmax(Q[env.state])
        this_episode.append([state, action])
        new_state, new_function_value, reward, done = env.step(action, time)
        if done:
            break
        Q_new[state, action] = Q[state, action] + alpha * (
                reward + gamma*np.max(Q[new_state]) - Q[state, action]
        )
        state = new_state
        difference = np.max(Q_new - Q)
        Q = np.copy(Q_new)
        time += timestep

    if episode % print_episode:
        print('Max difference in Q : ', difference)
        print('Episode : ', episode)

# def perform_generalized_policy_iteration():
#     global print_epoch, gamma, NUM_ACTIONS, NUM_STATES, Q, \
#         policy_matrix, running_mean_matrix, MAX_EPISODE_LENGTH, env, all_episode_lists
#
#     epoch = 0
#     gamma = 0.9
#
#     while True:
#         epoch += 1
#         episode_list = list()
#         observation = env.reset(exploring_starts=True)
#         # observation is the current state and the current value
#         # which the agent observes
#
#         done = False
#         # max length of each episode is 1000
#         for _ in range(MAX_EPISODE_LENGTH):
#             action = policy_matrix[observation[0]]
#
#             # Move one step and get a new observation and the reward
#             new_state, new_value, reward, done = env.step(action)
#             new_observation = [new_state, new_value]
#
#             # append what had you observed, and what action did you take resulting in what reward
#             episode_list.append((observation, action, reward))
#             observation = new_observation
#             if done:
#                 break
#
#         # For debugging
#         all_episode_lists.append(episode_list)
#
#         # This cycle is the implementation of First-Visit MC.
#         first_visit_done = np.zeros((NUM_ACTIONS, NUM_STATES))
#         counter = 0
#         # For each state-action stored in the episode list it checks if
#         # it is the first visit and then estimates the return.
#         # This is the Evaluation step of the GPI.
#         old_state_action_matrix = Q.copy()
#         for visit in episode_list:
#             state = visit[0][0]
#             action = int(visit[1])
#             if first_visit_done[action, state] == 0:
#                 return_value = get_return(episode_list[counter:], gamma)
#                 running_mean_matrix[action, state] += 1
#                 Q[action, state] += return_value
#                 first_visit_done[action, state] = 1
#             counter += 1
#         # Policy update (Improvement)
#
#         if has_converged(old_state_action_matrix/running_mean_matrix, Q / running_mean_matrix):
#             break
#
#         policy_matrix = update_policy(episode_list, policy_matrix, Q / running_mean_matrix)
#
#         if epoch % print_epoch == 0:
#             print("State-Action matrix after " + str(epoch) + " iterations:")
#             print(Q / running_mean_matrix)
#             print("Policy matrix after " + str(epoch + 1) + " iterations:")
#             print(policy_matrix)
#             describe_policy_matrix(policy_matrix, env)
#
#     # print("Utility matrix after " + str(N_EPOCHS) + " iterations: ")
#     # print(state_action_matrix/running_mean_matrix)
#     # print('Current Learnt Policy is ')
#     # describe_policy_matrix(policy_matrix)