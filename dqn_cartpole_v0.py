import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.display import clear_output, HTML
from tqdm.autonotebook import tqdm
import gym

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# In[]:

import matplotlib
# matplotlib.use('Qt5Agg')
cp_env = gym.make('CartPole-v1')


def plot_timesteps_and_rewards(avg_history):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.subplots_adjust(wspace=0.5)
    axes[0].plot(avg_history['episodes'], avg_history['timesteps'])
    axes[0].set_title('Timesteps in episode')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Timesteps')
    axes[1].plot(avg_history['episodes'], avg_history['reward'])
    axes[1].set_title('Reward')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Reward')
    plt.show()

def cp_run_current_policy(env, policy):
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

# Define the policy and replay buffer
class DQNPolicy:
    def __init__(self, env, lr, epsilon, gamma, hidden_dim=24):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.gamma = gamma

        # build the Q network approximator
        # TODO : Convert this to a torch model
        self.q_model = Sequential()
        self.q_model.add(Dense(self.hidden_dim, input_dim=self.state_dim, activation='relu'))
        self.q_model.add(Dense(self.hidden_dim, activation='relu'))
        # the last layer is linear in the 2nd last layer's output and it gives a probability of each of the actions
        self.q_model.add(Dense(self.action_dim, activation='linear'))
        self.q_model.compile(loss='mse', optimizer=Adam(lr=self.lr))

    def select_action(self, cur_state):
        # epsilon greedy strategy
        if np.random.uniform(low=0.0, high=1.0) > self.epsilon:
            # get Q(cur_state, a) for all action a
            predictions = self.q_model.predict(cur_state)[0]

            # select action with max Q value
            return np.argmax(predictions)
        else:
            # else return a random action
            return self.env.action_space.sample()

    def update_policy(self, cur_states, actions, next_states, rewards, dones):
        # target doesnt change when its terminal, thus multiply with (1-done)
        # target = R(st-1, at-1) + gamma * max(a') Q(st, a')
        targets = rewards + np.multiply(1 - dones, self.gamma * (np.max(self.q_model.predict(next_states), axis=1)))

        # V(st-1, at-1)
        expanded_targets = self.q_model.predict(cur_states)

        expanded_targets[list(range(len(cur_states))), actions] = targets

        self.q_model.fit(cur_states, expanded_targets, epochs=1, verbose=False)

# In[]:

class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.max_size = max_size

        self.cur_states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.cur_states)

    def add(self, cur_state, action, next_state, reward, done):
        self.cur_states.append(cur_state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, sample_size=32):
        sample_transitions = {}
        if self.__len__() >= sample_size:
            # pick up only random 32 events from the memory
            indices = np.random.choice(self.__len__(), size=sample_size)
            sample_transitions['cur_states'] = np.array(self.cur_states)[indices]
            sample_transitions['actions'] = np.array(self.actions)[indices]
            sample_transitions['next_states'] = np.array(self.next_states)[indices]
            sample_transitions['rewards'] = np.array(self.rewards)[indices]
            sample_transitions['dones'] = np.array(self.dones)[indices]
        else:
            # if the current buffer size is not greater than 32 then pick up the entire memory
            sample_transitions['cur_states'] = np.array(self.cur_states)
            sample_transitions['actions'] = np.array(self.actions)
            sample_transitions['next_states'] = np.array(self.next_states)
            sample_transitions['rewards'] = np.array(self.rewards)
            sample_transitions['dones'] = np.array(self.dones)
        return sample_transitions

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
cp_run_current_policy(cp_env.env, cp_policy)

cp_train_episodes = 120
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
cp_run_current_policy(cp_env, cp_policy)
