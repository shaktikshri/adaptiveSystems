# Implementation of Finite State IRL as Andrew and Russell, https://ai.stanford.edu/~ang/papers/icml00-irl.pdf
# The cleaning robot's Reward function is recovered

import numpy as np

# In[]:
# form the T matrix
T_s_a_sbar = np.zeros(shape=(25, 4, 25))

# 5*5 gridworld same as the cleaning robot. With start at (5,5) and end at (4,4)
# 0.8 chance of moving in the correct direction, 0.1,0.1 chance of moving in the direction perpendicular to it
# If the bot hits the wall while moving, it says where it is
for row in range(5):
    for col in range(5):
        state_index = row*5 + col

        possible_next_states = [max(row-1, 0)*5 + col, # state towards up
                                row*5 + min(col+1, 4),  # state towards right
                                min(row+1,4)*5 + col, # state towards down
                                row*5 + max(col-1,0)] # state towards left
        for a in range(4): # action 0: Up, 1: Right, 2: Down, 3: Left
            prob = np.zeros((4))
            prob[a] = 0.8
            prob[(a - 1) % 4] = 0.1
            prob[(a + 1) % 4] = 0.1

            # visualize this with action UP
            T_s_a_sbar[state_index, a, possible_next_states[0]] += prob[0]     # state towards UP gets 0.8
            T_s_a_sbar[state_index, a, possible_next_states[1]] += prob[1]     # state to the right gets 0.1
            T_s_a_sbar[state_index, a, possible_next_states[2]] += prob[2]     # state to the down gets 0
            T_s_a_sbar[state_index, a, possible_next_states[3]] += prob[3]     # state to the left gets 0.1

# In[]:
def print_policy(q_function):
    shape = q_function.shape
    policy_string = ''
    for row in range(shape[0]):
        for col in range(shape[1]):
            action = np.argmax(q_function[row, col, :])
            if(action == 0): policy_string += " ^  "
            elif(action == 1): policy_string += " >  "
            elif(action == 2): policy_string += " v  "
            elif(action == 3): policy_string += " <  "
        policy_string += '\n'
    print(policy_string)

# In[]:
# With this Transition Matrix we need to find out the best policy
# policy_list = list()
# for ntimes in range(10):

reward = np.full((5,5), -0.04)
reward[-1, -1] = 1
reward[-2, -1] = -3
Q = np.random.random((5,5,4))
Q_new = Q.copy()
gamma = 0.9
epsilon = 0.1
difference = 100
iterations = 0
utility = dict()
for el in range(25):
    utility.update({el:list()})
max_abs_diff = list()
min_abs_diff = list()
while difference > 0.0005:
    count = 0
# while iterations <= 10:
    iterations += 1
    for row in range(5):
        for col in range(5):
            # TODO : need to vectorize this
            for a in range(4):
                summation = 0
                for row_bar in range(5):
                    for col_bar in range(5):
                        summation += T_s_a_sbar[row*5+col, a, row_bar*5+col_bar]*np.max(Q[row_bar, col_bar, :])
                Q_new[row, col, a] = reward[row, col] + gamma * summation

            if row*5+col == 0:
                count += 1
            utility[row*5+col].append(np.max(Q_new[row, col, :]))
    max_abs_diff.append(np.max(np.absolute(Q-Q_new)))
    min_abs_diff.append(np.min(np.absolute(Q-Q_new)))
    print('Max Absolute Difference : ', np.max(np.absolute(Q-Q_new)))
    print('Min Absolute Difference : ', np.min(np.absolute(Q - Q_new)))
    difference = np.sum(np.absolute(Q - Q_new))
    Q = Q_new.copy()

# import matplotlib.pyplot as plt
# plt.figure(1)
# for el in range(25):
#     plt.plot(utility[el])
# plt.show()

print_policy(Q)
# policy_list.append(Q.argmax(axis=2))

# if np.all([np.all(policy_list[el] == policy_list[(el+1)%10]) for el in range(10)]):
#     print('Consistent Policy')
# else:
#     print('DUDE, something is seriously wrong')

# In[]:
# Implementation of Finite State Space IRL.
# See the paper Algorithms for Inverse Reinforcement Learning Section 3

# Flattening the Q matrix to be used
V_to_be_used = np.max(Q, axis=2).flatten().reshape(25, -1)
# V_to_be_used is a 25*1 vector having the utility of each of the state

# Reducing the 3D matrix T_s_a_sbar to a 2D matrix T_s_sbar so that we can
# efficiently use it in our vector multiplications
policy = Q.argmax(axis=2)
T_to_be_used = np.array([T_s_a_sbar[s, policy.flatten()[s]] for s in range(25)])
# T_to_be_used is a 25*25 2d matrix, denoting the probability from each state to the next state under the optimal policy
P_a1 = T_to_be_used

# P_a1 is 25*25
# To get P_a we can pick up the 2nd best action from the policy
best_policy = Q.argmax(axis=2)
# Now need to remove the Q values at these best indices, thus set them to some large negative value.
# and then take the argmax again to get the indices of the 2nd largest Q value, or the 2nd best action

value_to_be_replaced = Q.min() - 10 # this value can be replaced in place of the maximum Q values
Q_dummy = Q.copy()
Q_dummy[Q.max(axis=2).reshape(5,5,-1) == Q] = value_to_be_replaced
next_highest_values = Q_dummy.argmax(axis=2)
Ta_to_be_used = np.array([T_s_a_sbar[s, next_highest_values.flatten()[s]] for s in range(25)])
# Ta_to_be_used is a 25*25 2d matrix, denoting the probability from each state to the next state under the
# next best action for each of the state
P_a = Ta_to_be_used


# TODO: Need to check if this implementation of Pa is correct or not
# all_other_as = policy.flatten()
# for a in range(4):
#     P_a = np.array([T_s_a_sbar[s, a] for s in range(25) if policy.flatten()[s] != a])



# Need to formulate an linear program
import pulp
# TODO : we'll have to find out the critical gamma as well, the point after which all reward tend to be zero
gamma = 0.2

# Get a random reward
# TODO : we'll need to check if a random function can be provided while optimizing the Linear Program
R = np.random.random((25,1))

# the objective function is,
for i in range(25):
    # TODO : Axis has to be decided
    np.min((P_a1[i] - P_a[i])*(np.dot(np.linalg.inv(np.identity(P_a1.shape[0]) - gamma*P_a1)), R), axis=2)