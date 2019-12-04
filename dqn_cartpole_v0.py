import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from IPython.display import clear_output, HTML
from time import sleep
from tqdm.autonotebook import tqdm
import gym

# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib
matplotlib.use('Qt5Agg')
cp_env = gym.make('CartPole-v1')
cp_env.reset()
print("Observation space = {}".format(cp_env.observation_space))
print("Action space = {}".format(cp_env.action_space))

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
        targets = rewards + np.multiply(1 - dones, self.gamma * (np.max(self.q_model.predict(next_states), axis=1)))
        expanded_targets = self.q_model.predict(cur_states)

        # TODO : have a doubt here!
        expanded_targets[list(range(len(cur_states))), actions] = targets

        self.q_model.fit(cur_states, expanded_targets, epochs=1, verbose=False)


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
            indices = np.random.choice(self.__len__(), size=sample_size)
            sample_transitions['cur_states'] = np.array(self.cur_states)[indices]
            sample_transitions['actions'] = np.array(self.actions)[indices]
            sample_transitions['next_states'] = np.array(self.next_states)[indices]
            sample_transitions['rewards'] = np.array(self.rewards)[indices]
            sample_transitions['dones'] = np.array(self.dones)[indices]
        else:
            sample_transitions['cur_states'] = np.array(self.cur_states)
            sample_transitions['actions'] = np.array(self.actions)
            sample_transitions['next_states'] = np.array(self.next_states)
            sample_transitions['rewards'] = np.array(self.rewards)
            sample_transitions['dones'] = np.array(self.dones)
        return sample_transitions