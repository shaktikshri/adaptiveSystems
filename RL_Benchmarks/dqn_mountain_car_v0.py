from dqn import DQNPolicy, ReplayBuffer
import gym


def run_current_policy(policy, env, cur_state, epsilon):
    done = False
    total_reward = 0
    timesteps = 0
    while not done:
        action = policy.select_action(cur_state.reshape(1,-1), epsilon)
        next_state, reward, done, info = env.step(action)
        env.render()
        print('reward : ',reward),
        total_reward += reward
        timesteps += 1
        cur_state = next_state
    print('{} timesteps taken and collected {} reward'.format(timesteps, total_reward))
    return total_reward, timesteps

# In[]:

env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')

# TODO : Can change these parameters
lr = 0.001
# TODO : Need to do the epsilon decay
epsilon = 1
epsilon_decay = 0.05
epsilon_min = 0.01
gamma = 0.99
hidden_dim = 24
mod_episode = 10

env_policy = DQNPolicy(env, lr, gamma, hidden_dim)
replay_buffer = ReplayBuffer()
total_train_episodes = 500

# play with a random policy
# run_current_policy(env_policy, env, env.reset())

# In[]:
history = dict({'reward':list(), 'timesteps':list(), 'episodes':list()})

for episode in range(total_train_episodes):
    done = False
    # print('Epoch :', episode + 1)
    ep_reward = 0
    ep_timesteps = 0
    cur_state = env.reset()
    epsilon = max(epsilon, epsilon_min)
    max_position = -99
    while not done:
        action = env_policy.select_action(cur_state.reshape(1, -1), epsilon)
        next_state, reward, done, _ = env.step(action)

        # Visualize the status
        if episode % mod_episode == 0:
            env.render()

        # Keep track of max position
        if next_state[0] > max_position:
            max_position = next_state[0]

        # Adjust reward for task completion
        if next_state[0] >= 0.5:
            reward += 10

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
        print('Epoch : {} Success : {} Avg Reward : {} Timesteps : {} Max position : {}'.format(
            episode, max_position >= 0.5, history['reward'][-1], history['timesteps'][-1], max_position))

    # decay the epsilon after every episode
    epsilon -= epsilon_decay

# In[]:

# Now play again
run_current_policy(env_policy, env, env.reset(), epsilon)
env.close()

# In[]:
import matplotlib
matplotlib.use('Qt5Agg')
from plot_functions import plot_timesteps_and_rewards
plot_timesteps_and_rewards(history)