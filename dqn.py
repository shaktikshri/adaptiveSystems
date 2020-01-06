from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import torch


# Define the policy and replay buffer
class DQNPolicy:
    def __init__(self, env, lr, gamma, hidden_dim=24, input=None, output=None):
        self.env = env
        if input is None:
            self.state_dim = env.observation_space.shape[0]
        else:
            self.state_dim = input
        if output is None:
            self.action_dim = env.action_space.n
        else:
            self.action_dim = output
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # build the Q network approximator
        # TODO : Convert this to a torch model
        self.q_model = Sequential()
        self.q_model.add(Dense(self.hidden_dim, input_dim=self.state_dim, activation='relu'))
        self.q_model.add(Dense(self.hidden_dim, activation='relu'))
        # the last layer is linear in the 2nd last layer's output and it gives a probability of each of the actions
        self.q_model.add(Dense(self.action_dim, activation='linear'))
        self.q_model.compile(loss='mse', optimizer=Adam(lr=self.lr))

    def select_action(self, cur_state, epsilon):
        # epsilon greedy strategy
        if np.random.uniform(low=0.0, high=1.0) > epsilon:
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

        # expanded_targets are the Q values of all the actions for the current_states sampled
        # from the previous experience. These are the predictions
        expanded_targets = self.q_model.predict(cur_states)

        # Prediction to be updated with the prediction+ground truth
        # We need to update the predictions to the values we want, which are the targets and then fit the model
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

    def sample_pytorch(self, sample_size=32):
        sample_transitions = {}
        if self.__len__() >= sample_size:
            # pick up only random 32 events from the memory
            indices = np.random.choice(self.__len__(), size=sample_size)
            sample_transitions['cur_states'] = torch.stack(self.cur_states)[indices]
            sample_transitions['actions'] = torch.stack(self.actions)[indices]
            sample_transitions['next_states'] = torch.stack(self.next_states)[indices]
            sample_transitions['rewards'] = torch.Tensor(self.rewards)[indices]
            sample_transitions['dones'] = torch.Tensor(self.dones)[indices]
        else:
            # if the current buffer size is not greater than 32 then pick up the entire memory
            sample_transitions['cur_states'] = torch.stack(self.cur_states)
            sample_transitions['actions'] = torch.stack(self.actions)
            sample_transitions['next_states'] = torch.stack(self.next_states)
            sample_transitions['rewards'] = torch.Tensor(self.rewards)
            sample_transitions['dones'] = torch.Tensor(self.dones)

        return sample_transitions
