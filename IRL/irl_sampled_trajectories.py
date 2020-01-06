# this is learning reward function from a sampled trajectory
import numpy as np

# In[]:

# constructing a 5*5 gridworld as put by Andrew and Russel in Inverse Reinforcement Learning
# state is continuous [0,1]*[0,1]
class Agent:
    def __init__(self):
        self.cur_state = np.random.rand(2)
        # the action space is used for the policy selection in the deep q network
        self.action_space = self.obj()
    class obj:
        def __init__(self):
            pass
        def sample(self):
            return np.random.choice([0, 1, 2, 3])
    def step(self, action):
        assert action in [0, 1, 2, 3]
        # UP is 0, RIGHT 1, DOWN 2, LEFT 3
        if action == 0:
            step_size = np.array([0, 0.2])
        elif action == 1:
            step_size = np.array([0.2, 0])
        elif action == 2:
            step_size = np.array([0, -0.2])
        else:
            step_size = np.array([-0.2, 0])
        # whenever an action is performed, the guy moves in that direction and then a uniform noise from [-0.1, 0.1]
        # is added in each of the coordinates
        self.cur_state = self.cur_state + step_size + \
                         np.array([(2*np.random.rand() - 1)/10, (2*np.random.rand() - 1)/10])
        # Truncate the state to be in [0,1]*[0,1]
        self.cur_state[self.cur_state > 1] = 1
        self.cur_state[self.cur_state < 0] = 0
        return self.cur_state

"""
Remember whenever you're in lack of domain for basis functions, you can always take the domain to be the same
as the domain of the state space. And as always, Gaussian Mixtures are the best choice for basis functions
"""
# generate evenly spaced 15*15 2d gaussian over the state space
cov = [[0.1, 0],[0, 0.1]]
std = 0.1
mean = np.arange(0, 1, 1/15)
from scipy.stats import multivariate_normal

basis = np.array([])
for i in range(15):
    for j in range(15):
        basis = np.append(basis, multivariate_normal(mean=[mean[i], mean[j]], cov=cov))
basis = basis.reshape(15, 15)

reward_function = lambda state: 1 if np.all(state >= np.array([0.8, 0.8])) else 0
# visualize the true reward distribution
# from plot_functions import figure
# x_points = np.arange(0, 1, 0.01)
# z = np.zeros((100, 100))
# for i in range(100):
#     for j in range(100):
#         z[i, j] = reward(np.array([x_points[i], x_points[j]]))
# figure(x_points, x_points, z, title='true reward')

env = Agent()

# In[]:
# We need a capability to find the optimal policy for any given reward distribution
"""This is where the algorithm is expensive, it needs you to find optimal policies for intermediate reward distributions
Now since the state space is continuous, any method– either discretization or Q learning will be expensive """
# Do a Q learning to learn the optimal policy for the reward structure given
# Using Q Learning to learn the optimal policy as per the reward distribution

from dqn import DQNPolicy, ReplayBuffer
from plot_functions import plot_timesteps_and_rewards
alpha = 0.01
gamma = 0.9
epsilon = 0.1
policy = DQNPolicy(env, lr=alpha, gamma=gamma, input=1, output=4) # 4 actions output, up, right, down, left
replay_buffer = ReplayBuffer()
start_episode = 0
avg_reward = 0
avg_timestep = 0
# Play with a random policy and see
# run_current_policy(env.env, policy)
train_episodes = 200
agg_interval = 10
avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
# Train the network to predict actions for each of the states
for episode_i in range(start_episode, start_episode + train_episodes):
    episode_timestep = 0
    episode_reward = 0.0
    env.__init__()
    cur_state = env.cur_state
    counter = 0
    done = False
    while not done:
        # Let each episode be of 30 steps
        counter += 1
        done = counter >= 30

        # todo : check if this line is working
        action = policy.select_action(cur_state.reshape(1, -1), epsilon)

        # take action in the environment
        next_state = env.step(action)
        reward = reward_function(next_state)

        # add the transition to replay buffer
        replay_buffer.add(cur_state, action, next_state, reward, done)

        # sample minibatch of transitions from the replay buffer
        # the sampling is done every timestep and not every episode
        sample_transitions = replay_buffer.sample()

        # update the policy using the sampled transitions
        policy.update_policy(**sample_transitions)

        episode_reward += reward
        episode_timestep += 1

        cur_state = next_state

    avg_reward += episode_reward
    avg_timestep += episode_timestep

    if (episode_i + 1) % agg_interval == 0:
        avg_history['episodes'].append(episode_i + 1)
        avg_history['timesteps'].append(avg_timestep / float(agg_interval))
        avg_history['reward'].append(avg_reward / float(agg_interval))
        avg_timestep = 0
        avg_reward = 0.0

plot_timesteps_and_rewards(avg_history)

# In[]:
