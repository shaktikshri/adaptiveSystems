# Implementation of Finite State IRL as Andrew and Russell, https://ai.stanford.edu/~ang/papers/icml00-irl.pdf
# The cleaning robot's Reward function is recovered

import numpy as np

# In[]:
ROWS = 2
COLUMNS = 2
ACTIONS = 4
# form the T matrix
T_s_a_sbar = np.zeros(shape=(ROWS*COLUMNS, ACTIONS, ROWS*COLUMNS))

# ROWS*COLUMNS gridworld same as the cleaning robot. With start at (0,0) and end at (ROWS-1,COLUMNS-1)
# 0.8 chance of moving in the correct direction, 0.1,0.1 chance of moving in the direction perpendicular to it
# If the bot hits the wall while moving, it says where it is
for row in range(ROWS):
    for col in range(COLUMNS):
        state_index = row*COLUMNS + col

        possible_next_states = [max(row-1, 0)*COLUMNS + col, # state towards up
                                row*COLUMNS + min(col+1, COLUMNS-1),  # state towards right
                                min(row+1,ROWS-1)*COLUMNS + col, # state towards down
                                row*COLUMNS + max(col-1,0)] # state towards left
        for a in range(ACTIONS): # action 0: Up, 1: Right, 2: Down, 3: Left
            prob = np.zeros((ACTIONS))
            prob[a] = 0.8
            prob[(a - 1) % ACTIONS] = 0.1
            prob[(a + 1) % ACTIONS] = 0.1

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

reward = np.full((ROWS,COLUMNS), -0.04)
reward[-1, -1] = 1
reward[-2, -1] = -3
Q = np.random.random((ROWS,COLUMNS,ACTIONS))
Q_new = Q.copy()
gamma = 0.9
epsilon = 0.1
difference = 100
iterations = 0
utility = dict()
for el in range(ROWS*COLUMNS):
    utility.update({el:list()})
max_abs_diff = list()
min_abs_diff = list()
while difference > 0.0005:
    count = 0
# while iterations <= 10:
    iterations += 1
    for row in range(ROWS):
        for col in range(COLUMNS):
            # TODO : need to vectorize this
            for a in range(ACTIONS):
                summation = 0
                for row_bar in range(ROWS):
                    for col_bar in range(COLUMNS):
                        summation += T_s_a_sbar[row*COLUMNS+col, a, row_bar*COLUMNS+col_bar]*np.max(Q[row_bar, col_bar, :])
                Q_new[row, col, a] = reward[row, col] + gamma * summation

            if row*COLUMNS+col == 0:
                count += 1
            utility[row*COLUMNS+col].append(np.max(Q_new[row, col, :]))
    max_abs_diff.append(np.max(np.absolute(Q-Q_new)))
    min_abs_diff.append(np.min(np.absolute(Q-Q_new)))
    print('Max Absolute Difference : ', np.max(np.absolute(Q-Q_new)))
    print('Min Absolute Difference : ', np.min(np.absolute(Q - Q_new)))
    difference = np.sum(np.absolute(Q - Q_new))
    Q = Q_new.copy()

# import matplotlib.pyplot as plt
# plt.figure(1)
# for el in range(ROWS*COLUMNS):
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
V_to_be_used = np.max(Q, axis=2).flatten().reshape(ROWS*COLUMNS, -1)
# V_to_be_used is a ROWS*COLUMNS*1 vector having the utility of each of the state

# Reducing the 3D matrix T_s_a_sbar to a 2D matrix T_s_sbar so that we can
# efficiently use it in our vector multiplications
policy = Q.argmax(axis=2)
T_to_be_used = np.array([T_s_a_sbar[s, policy.flatten()[s]] for s in range(ROWS*COLUMNS)])
# T_to_be_used is a ROWS*COLUMNS*ROWS*COLUMNS 2d matrix, denoting the probability from each state to the next state under the optimal policy
P_a1 = T_to_be_used

# P_a1 is ROWS*COLUMNS*ROWS*COLUMNS
# To get P_a we can pick up the 2nd best action from the policy
best_policy = Q.argmax(axis=2)
# Now need to remove the Q values at these best indices, thus set them to some large negative value.
# and then take the argmax again to get the indices of the 2nd largest Q value, or the 2nd best action

value_to_be_replaced = Q.min() - 10 # this value can be replaced in place of the maximum Q values
Q_dummy = Q.copy()
Q_dummy[Q.max(axis=2).reshape(ROWS,COLUMNS,-1) == Q] = value_to_be_replaced
next_highest_values = Q_dummy.argmax(axis=2)
Ta_to_be_used = np.array([T_s_a_sbar[s, next_highest_values.flatten()[s]] for s in range(ROWS*COLUMNS)])
# Ta_to_be_used is a ROWS*COLUMNS*ROWS*COLUMNS 2d matrix, denoting the probability from each state to the next state under the
# next best action for each of the state
P_a = Ta_to_be_used

# Get a random reward
Rmax = 10
Rmin = 0

# In[]:
from pulp import *
import matplotlib.pyplot as plt
# We are now ready to formulate this as a LinearProgram
for lambda_val in np.linspace(0,1.5,num=10):
    prob = LpProblem('IRL_Reward', LpMaximize)
    RANGE = range(ROWS*COLUMNS)
    R = LpVariable.dicts('R', RANGE)
    for i in R.keys():
         R[i].lowBound = Rmin
         R[i].upBound = Rmax

    complicated = np.linalg.inv(np.identity(P_a1.shape[0]) - gamma*P_a1)
    # We dont need min in the objective function because we have already found the next best action for each
    prob += np.sum([np.dot(P_a1[i] - P_a[i], np.dot(complicated, np.array([R[el1] for el1 in range(ROWS*COLUMNS)]).reshape(ROWS*COLUMNS,1)))
                    - lambda_val*lpSum([R[el2] for el2 in range(ROWS*COLUMNS)])
                   for i in range(ROWS*COLUMNS) ])
    expression = np.dot(P_a1 - P_a, np.dot(complicated, np.array([R[el3] for el3 in range(ROWS*COLUMNS)]).reshape(ROWS*COLUMNS,1))) # "greater than or equal to 0 constraint"
    for el in expression:
        prob += el[0] >= 0

    prob.solve()
    print('Lambda : ',lambda_val, [value(R[el]) for el in range(ROWS*COLUMNS)])
    plt.plot([value(R[el]) for el in range(ROWS*COLUMNS)], label='lambda : '+str(lambda_val))
plt.plot(reward.flatten(), '--', label='True Reward')
plt.legend()
