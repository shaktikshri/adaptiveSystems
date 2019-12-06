import numpy as np
x = np.arange(0,2*np.pi,0.01)

def f(x, noise):
    return np.sin(x) + noise

# f = lambda x: np.sin(x)
# import matplotlib.pyplot as plt
# noise = np.random.choice([0.025, -0.025, 0.05, -0.05], size=x.shape[0])
# states = np.array([x, f(x, noise)])
# plt.plot(states[0], states[1], label='original', color='b')

def step(states, actions):
    states[1] += actions
    return states

# actions = -noise
# states = step(states, actions)
# plt.plot(states[0], states[1], label='smoothened', color='g')
# plt.legend()
# plt.show()

# In[]:


class RandomVariable():
    class ActionSpace():
        def __init__(self):
            self.actions = np.linspace(-0.1, +0.1, 10)
            self.n = self.actions.shape[0]

        def sample(self):
            return np.random.choice(self.n)

    def __init__(self, errepsilon,  noise_levels, x_increment, x_range):
        self.y = 0
        self.x = 0
        self.x_increment = x_increment
        self.x_range = x_range
        self.noise_levels = noise_levels
        # TODO : Change these values and check
        self.errepsilon = errepsilon
        self.observation_space = np.array([2,2])
        self.action_space = self.ActionSpace()

    def step(self, action):
        self.y += action
        reward = self.get_reward()
        self.x = ( self.x + self.x_increment ) % self.x_range
        return np.array([self.x, self.f()]), reward

    def f(self):
        return np.sin(self.x) + np.random.choice(self.noise_levels)

    def get_reward(self):
        if np.abs(self.f() - self.y) < self.errepsilon:
            # TODO : Change the reward values and check
            return +10
        else:
            return -1

    def reset(self):
        self.x = 0
        self.y = self.f()
        return np.array([self.x, self.y])


# In[]:

from dqn import DQNPolicy, ReplayBuffer


def run_current_policy(policy, env, cur_state, epsilon, max_iterations):
    total_reward = 0
    function_history = list()
    timesteps = 0
    for iterations in range(max_iterations):
        action = policy.select_action(cur_state.reshape(1,-1), epsilon)
        next_state, reward = env.step(action)
        function_history.append(cur_state)
        total_reward += reward
        timesteps += 1
        cur_state = next_state
    print('{} timesteps taken and collected {} reward'.format(timesteps, total_reward))
    return total_reward, timesteps, np.array(function_history)

# In[]:

noise = [ -0.07777778, 0.07777778]


# TODO : Can change these parameters
lr = 0.001
# TODO : Need to do the epsilon decay
epsilon = 1
epsilon_decay = 0.05
epsilon_min = 0.01
gamma = 0.99
hidden_dim = 50
mod_episode = 10
max_iterations = 500
x_range = 10
x_increment = 0.01
max_x = x_increment * max_iterations

env = RandomVariable(0.001, noise, x_increment, x_range)
env_policy = DQNPolicy(env, lr, gamma, hidden_dim)
replay_buffer = ReplayBuffer()
total_train_episodes = 500

# play with a random policy
# run_current_policy(env_policy, env, env.reset(), max_iterations)

# In[]:
history = dict({'reward':list(), 'timesteps':list(), 'episodes':list()})

import matplotlib.pyplot as plt

plt.ion()

fig, ax = plt.subplots()
noise_pl = np.random.choice([0.025, -0.025, 0.05, -0.05], size=x.shape[0])
states_pl = np.array([x, f(x, noise_pl)])
sc = ax.scatter(states_pl[0], states_pl[1])
plt.xlim(0, max_x)
plt.ylim(-1, 1)
plt.draw()


for episode in range(1, total_train_episodes):
    done = False
    # print('Epoch :', episode + 1)
    ep_reward = 0
    ep_timesteps = 0
    cur_state = env.reset()
    epsilon = max(epsilon, epsilon_min)
    for iterations in range(max_iterations):
        action = env_policy.select_action(cur_state.reshape(1, -1), epsilon)
        next_state, reward = env.step(action)

        replay_buffer.add(cur_state, action, next_state, reward, done)

        # TODO : Change the sample size and check any improvements
        sampled_transitions = replay_buffer.sample()
        # the q updation occurs for all transitions in all episodes, just like TD updates
        env_policy.update_policy(**sampled_transitions)
        ep_reward += reward
        ep_timesteps += 1

        cur_state = next_state

    history['reward'].append(ep_reward)
    history['timesteps'].append(ep_timesteps)
    history['episodes'].append(episode+1)
    if episode % mod_episode == 0:
        # Get last 100 points from replay buffer
        states = np.array(replay_buffer.cur_states[:-100])
        print('Epoch : {} Avg Reward : {} Timesteps : {}'.format(
            episode, history['reward'][-1], history['timesteps'][-1]))
        # plt.figure()
        sc.set_offsets(np.c_[states[:, 0], states[:, 1]])
        fig.canvas.draw_idle()
        plt.pause(0.1)

    # decay the epsilon after every episode
    epsilon -= epsilon_decay

plt.ioff()
plt.show()

# In[]:

# Now play again
_, _, states = run_current_policy(env_policy, env, env.reset(), epsilon, max_iterations)
plt.scatter(states[:, 0], states[:, 1])

# In[]:
# import matplotlib
# matplotlib.use('Qt5Agg')
from plot_functions import plot_timesteps_and_rewards
plot_timesteps_and_rewards(history)