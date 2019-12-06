import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import gym
from dqn import DQNPolicy, ReplayBuffer
from plot_functions import plot_timesteps_and_rewards

# In[]:

import matplotlib
# matplotlib.use('Qt5Agg')
cp_env = gym.make('CartPole-v1')


def run_current_policy(env, policy):
    cur_state = env.reset()
    total_step = 0
    total_reward = 0.0
    done = False
    while not done:
        action = policy.select_action(np.array([cur_state]))
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        env.render(mode='rgb_array')
        total_step += 1
        cur_state = next_state
    print("Total timesteps = {}, total reward = {}".format(total_step, total_reward))

# In[]:

cp_alpha = 0.001
cp_gamma = 0.95
cp_epsilon = 0.05
cp_avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
agg_interval = 1
avg_reward = 0.0
avg_timestep = 0

# initialize policy and replay buffer
cp_policy = DQNPolicy(cp_env, lr=cp_alpha, epsilon=cp_epsilon, gamma=cp_gamma)
replay_buffer = ReplayBuffer()
cp_start_episode = 0

# Play with a random policy and see
# run_current_policy(cp_env.env, cp_policy)

cp_train_episodes = 200
pbar_cp = tqdm(total=cp_train_episodes)

# In[]:

# Train the network to predict actions for each of the states
for episode_i in range(cp_start_episode, cp_start_episode + cp_train_episodes):
    episode_timestep = 0
    episode_reward = 0.0

    done = False

    cur_state = cp_env.reset()

    while not done:
        # select action
        action = cp_policy.select_action(cur_state.reshape(1, -1))

        # take action in the environment
        next_state, reward, done, info = cp_env.step(action)

        # add the transition to replay buffer
        replay_buffer.add(cur_state, action, next_state, reward, done)

        # sample minibatch of transitions from the replay buffer
        # the sampling is done every timestep and not every episode
        sample_transitions = replay_buffer.sample()

        # update the policy using the sampled transitions
        cp_policy.update_policy(**sample_transitions)

        episode_reward += reward
        episode_timestep += 1

        cur_state = next_state

    avg_reward += episode_reward
    avg_timestep += episode_timestep

    if (episode_i + 1) % agg_interval == 0:
        cp_avg_history['episodes'].append(episode_i + 1)
        cp_avg_history['timesteps'].append(avg_timestep / float(agg_interval))
        cp_avg_history['reward'].append(avg_reward / float(agg_interval))
        avg_timestep = 0
        avg_reward = 0.0

    pbar_cp.update()
cp_start_episode = cp_start_episode + cp_train_episodes
plot_timesteps_and_rewards(cp_avg_history)
run_current_policy(cp_env, cp_policy)
cp_env.close()
