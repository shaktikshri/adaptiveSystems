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
print('Utility of state (1,1) = ',utility_11)
