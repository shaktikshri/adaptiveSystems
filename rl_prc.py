import numpy as np

T = np.array([
    [0.9, 0.1],
    [0.5, 0.5]
])

# declaring initial distribution
v = np.array(
    [1, 0]
)

for i in [1,2,3,4,5]:
    T_i = np.linalg.matrix_power(T, i)
    print('Transition Prob after time k=',i,'\n',T_i)
    vti = np.dot(v, T_i)
    print('Prob of being in a specific state after', i, 'iterations = \n', vti)
# the transition probability converges to array([
#           [0.83333333, 0.16666667],
#           [0.83333333, 0.16666667]
#     ])
# Thus the asymptotic belief is that after infinite steps, the state will be
# system will remain in s0 with a prob of 83.3% and move to s1 16.7% of the time
# Upon moving to s1, it will stay in s1 16.7% of the time, and move to s0
# with 83.3% probability
# this is ind. of the initial state as we can see with the initial state vector.
# thus this Markov process has converged.

# In[]:
# there are 12 states, thus the transition matrix should be a 12*12 matrix giving
# the transition prob. from each state to every other state
# Now there are 4 actions also, thus the transition matrix should have this also since
# the prob. of moving to other state is a function of T(s,s',a).
# Thus the transition prob will be a 12*12*4 matrix, where each T[:,:,0] is the transition
# matrix for action 0, and so on.
T = np.load('T.npy')
NUM_ACTIONS = 4

# the agent starts from position (1,1) which is the bottom-most left corner
v = np.array([[
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0
]])
# Utility vector : this utility vector came from the value iteration algorithm
u = np.array([[
    0.812, 0.868, 0.918, 1.0,
    0.762, 0.0, 0.660, -1.0,
    0.705, 0.655, 0.611, 0.388
]])
reward = -0.04
gamma = 1

# In[]:


def return_state_utility(v, T, u, reward, gamma):
    """Return the state utility.
    @param v the state vector
    @param T transition matrix
    @param u utility vector
    @param reward for that state
    @param gamma discount factor
    @return the utility of the state
    """
    action_array = np.zeros(NUM_ACTIONS)
    for action in range(NUM_ACTIONS):
        action_array[action] = np.sum(np.multiply(u, np.dot(v, T[:, :, action])))
    return reward + gamma * np.max(action_array)

# In[]:


utility_11 = return_state_utility(v, T, u, reward, gamma)
print('Utility of state (1,1) = ', utility_11)

# In[]:
# The Value iteration algorithm to compute the utility vector

tot_states = 12

# Discount Factor
gamma = 0.999
iteration = 0

#Stopping criteria small value
epsilon = 0.001

# List containing the data for each iteation
graph_list = list()

#Reward vector, note the +1 and -1 for the charging station and the stairs
r = np.array([-0.04, -0.04, -0.04, +1.0,
    -0.04, 0.0, -0.04, -1.0, -0.04, -0.04, -0.04, -0.04
])

# Initial Utility vectors
u1 = np.array([0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,  0.0
])

# In[]:

# The Value Iteration Algorithm
# You only need to define the initial transition probabilities wrt each action
# and wrt each state to state, and the rewards for each state.
# The utilities and everything else will be learnt
# from here
while True:
    delta = 0
    u = u1.copy()
    iteration += 1
    graph_list.append(u)

    for state in range(tot_states):
        reward = r[state]
        state_vector = np.zeros((1, tot_states))
        state_vector[0, state] = 1
        u1[state] = return_state_utility(state_vector, T, u, reward, gamma)
        delta = max(delta, np.abs(u1[state] - u[state]))

    if delta < epsilon * (1-gamma) / gamma:
        print("=================== FINAL RESULT ==================")
        print("Iterations: " + str(iteration))
        print("Delta: " + str(delta))
        print("Gamma: " + str(gamma))
        print("Epsilon: " + str(epsilon))
        print("===================================================")
        print('Utility Values')
        print(u[0:4])
        print(u[4:8])
        print(u[8:12])
        print("===================================================")
        break

# In[]:
# Plotting the utility value convergence of each state
import matplotlib.pyplot as plt
for state in range(tot_states):
    utility_vals_history = [el[state] for el in graph_list]
    plt.plot(utility_vals_history, label='state'+str(state))
plt.legend()
plt.show()
