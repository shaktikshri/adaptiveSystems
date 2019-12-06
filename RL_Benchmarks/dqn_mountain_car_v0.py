from dqn import DQNPolicy, ReplayBuffer
import gym
from tqdm.autonotebook import tqdm


def run_current_policy(policy, env, cur_state):
    done = False
    total_reward = 0
    timesteps = 0
    while not done:
        action = policy.select_action(cur_state.reshape(1,-1))
        next_state, reward, done, info = env.step(action)
        env.render()
        print('reward : ',reward),
        total_reward += reward
        timesteps += 1
    print('{} timesteps taken and collected {} reward'.format(timesteps, total_reward))
    return total_reward, timesteps

# In[]:

env = gym.make('MountainCar-v0')

# TODO : Can change these parameters
lr = 0.001
epsilon = 0.05
gamma = 0.9
hidden_dim = 100
mod_episode = 10

env_policy = DQNPolicy(env, lr, epsilon, gamma, hidden_dim)
replay_buffer = ReplayBuffer()
total_train_episodes = 150

# play with a random policy
# run_current_policy(env_policy, env, env.reset())

# In[]:
progress_bar = tqdm(total=total_train_episodes)
history = dict({'reward':list(), 'timesteps':list(), 'episodes':list()})

for episode in range(total_train_episodes):
    done = False
    # print('Epoch :', episode + 1)
    ep_reward = 0
    ep_timesteps = 0
    cur_state = env.reset()
    while not done:
        action = env_policy.select_action(cur_state.reshape(1, -1))
        next_state, reward, done, info = env.step(action)
        replay_buffer.add(cur_state, action, next_state, reward, done)

        # TODO : Change the sample size and check any improvements
        sampled_transitions = replay_buffer.sample()
        env_policy.update_policy(cur_states=sampled_transitions['cur_states'],
                                 actions=sampled_transitions['actions'],
                                 next_states=sampled_transitions['next_states'],
                                 rewards=sampled_transitions['rewards'],
                                 dones=sampled_transitions['dones'])
        # env_policy.update_policy(**sampled_transitions)
        ep_reward += reward
        ep_timesteps += 1
    history['reward'].append(ep_reward)
    history['timesteps'].append(ep_timesteps)
    history['episodes'].append(episode+1)
    if episode % mod_episode == 0:
        print('Epoch : ', episode)
        print('Avg Reward : ', history['reward'][-1])
        print('TimeSteps : ', history['timesteps'][-1])
    progress_bar.update()

# In[]:

# Now play again
run_current_policy(env_policy, env, env.reset())
env.close()

# In[]:
from plot_functions import plot_timesteps_and_rewards
plot_timesteps_and_rewards(history)