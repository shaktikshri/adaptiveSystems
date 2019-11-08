import numpy as np
import matplotlib.pyplot as plt


def f(x, mean, std):
    return np.sin(x) + np.random.normal(loc=mean, scale=std)


total_size = 1000
a = list()
value = f(0, 0, 0.5)
for el in range(total_size):
    a.append(value)
    value = f(value, 0, 0.5)

a = np.array(a)
plt.plot(np.arange(1,total_size+1), a)
plt.xlim(100, 200)

# In[]:


class RandomVariable:
    def __init__(self):
        self.mean = 0
        self.std = 0.5
        self.state = 0
        self.curr_val = 0
        self.safe_mod = 3
        self.unsafe_mod = 5

    def reset(self, exploring_starts):
        if exploring_starts:
            self.state = np.random.choice(3, 1)
        else:
            self.state = 0

    def evaluate_state(self):
        # defined 3 states, safe (0), unsafe (1), critical (2)
        # Safe : value hops between [-3, +3]
        # unsafe : value between [-5,-3) and (+3, +5]
        # critical : values beyond that
        if abs(self.curr_val) < self.safe_mod:
            self.state = 0
        elif abs(self.curr_val) < self.unsafe_mod:
            self.state = 1
        self.state = 2

    def f(self, x, mean, std):
        return np.sin(x) + np.random.normal(loc=mean, scale=std)

    def get_value(self, mean=None, std=None):
        if not mean and not std:
            return self.f(self.curr_val, self.mean, self.std)
        else:
            return self.f(self.curr_val, mean, std)

    def step(self, action):
        self.curr_val += action
        self.evaluate_state()
        return self.state, env.curr_val


obj = RandomVariable()
y = list()
obj.get_value()
obj.get_state()





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
        if policy_matrix[observation] != -1:
            # if its not the terminal state
            policy_matrix[observation] = np.argmax(state_action_matrix[:, None])
    return policy_matrix


def get_reward(action, prev_value):
    """
    Returns the reward for a particular action taken on observing a
    particular metric value prev_value
    :param action: the action taken in this state
    :param prev_value: the value of the metric when the action was taken
    :return: the reward
    """
    if abs(prev_value + action) < 3:
        return +1
    elif abs(prev_value + action) < 5:
        return -0.4
    else:
        return -2


env = RandomVariable()
gamma = 0.99
print_epoch = 10000

state_matrix = np.zeros((3,))
# this is the safe state
state_matrix[0] = 1

reward = np.zeros((3,))
reward[0] = +1
reward[1] = -0.5
reward[2] = -2

# Random policy matrix
policy_matrix = np.random.randint(low=0, high=4, size=(3,)).astype(np.float32)

# State-action matrix or the Q values (init to zeros or to random values)
state_action_matrix = np.random.random_sample((4, 12))
running_mean_matrix = np.full((4, 12), 1.0e-12)
# one row of all states for each action, thus 12 columns for each row
n_epochs = 500000

for epoch in range(n_epochs):
    episode_list = list()
    env.reset(exploring_starts=True)

    # observation is the current state and the current value
    # which the agent observes
    observation = [env.state, env.curr_val]

    is_starting = True
    done = False
    # length of each episode is 1000
    for _ in range(1000):
        action = policy_matrix[observation[0]]

        # Move one step and get a new observation and the reward
        new_observation = env.step(action)
        if new_observation == 0:
            done = True

        # get the reward when you took action 'action' upon
        # observing the metric value
        reward = get_reward(observation[1], action)

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

print("Utility matrix after " + str(n_epochs) + " iterations: ")
print(state_action_matrix)
