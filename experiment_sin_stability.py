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

NUM_STATES = 7
NUM_ACTIONS = 7
MAX_EPISODE_LENGTH = 1000
N_EPOCHS = 500000


class RandomVariable:
    def __init__(self):
        self.mean = 0
        self.std = 0.5
        self.state = 0
        self.curr_val = 0
        self.safe_mod = 3
        self.unsafe_mod = 5
        self.critical_mod = 10

    def reset(self, exploring_starts):
        if exploring_starts:
            # Randomly select a current value
            low, high = -self.critical_mod-2, self.critical_mod+2
        else:
            low, high = -self.safe_mod-2, self.safe_mod+2
        self.curr_val = np.random.uniform(low=low, high=high)
        # Set the state according to the metric
        self.evaluate_state()

    def evaluate_state(self):
        """
        defined 7 states,
        (State 3)       Safe : value hops between [-3, +3]
        (State 2 and 4) unsafe : value between [-5,-3) and (+3, +5]
        (State 1 and 5) critical : values between [-10, -5) and (+5, +10]
        (State 0 and 6) incident : values between (-inf, -10) and (+10, +inf)
        Out of these incident and safe are the terminal states
        the episode will end at safe (meaning the value has stabilized) or
        the episode will end at incident (meaning there is nothing you can do
        to stabilize the value in the incident state, and you failed)
        :return:
        """
        if abs(self.curr_val) < self.safe_mod:
            self.state = 0
        elif abs(self.curr_val) < self.unsafe_mod:
            self.state = 2 if self.curr_val<0 else 4
        elif abs(self.curr_val) < self.critical_mod:
            self.state = 1 if self.curr_val<0 else 5
        else:
            self.state = 0 if self.curr_val<0 else 6

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
obj.evaluate_state()




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


def get_reward(prev_value, action):
    """
    Returns the reward for a particular action taken on observing a
    particular metric value prev_value
    :param prev_value: the value of the metric when the action was taken
    :param action: the action taken in this state
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

# There are 7 possible actions
# In State 0: Do nothing, add 0 since you are dead!
# In State 1: Add 10 to the metric
# In State 2: Add 5 to the metrix
# In State 3: Do Nothing, enjoy the peace!
# In State 4: Subtract 5 from the metric
# In State 5: Subtract 10 from the metric
# In State 6: Do nothing, add 0 since you are dead!
action_matrix = np.zeros(NUM_ACTIONS, 1)

# Random policy matrix
policy_matrix = np.random.randint(low=0, high=NUM_ACTIONS, size=(NUM_STATES,)).astype(np.float32)

# State-action matrix or the Q values (init to zeros or to random values)
state_action_matrix = np.random.random_sample((NUM_ACTIONS, NUM_STATES))
running_mean_matrix = np.full((NUM_ACTIONS, NUM_STATES), 1.0e-12)
# one row of all states for each action, thus NUM_STATES columns for each row


for epoch in range(N_EPOCHS):
    episode_list = list()
    env.reset(exploring_starts=True)

    # observation is the current state and the current value
    # which the agent observes
    observation = [env.state, env.curr_val]

    done = False
    # max length of each episode is 1000
    for _ in range(MAX_EPISODE_LENGTH):
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