import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import gym
from dqn import ReplayBuffer
from torch.distributions import Categorical

# TODO : 
#  1. Experience replay 
#  2. Fixing target
#  3. Learning rate decay with a scheduler


class FunctionApproximation(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=12):
        super(FunctionApproximation, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            # TODO : Try out log here if any numerical instability occurs
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.output_layer(out)
        return out

    def select_action(self, current_state):
        """
        selects an action as per some decided exploration
        :param current_state: the current state
        :return: the chosen action and the log probility of that chosen action
        """
        probs = self(current_state)
        # TODO : This can be made as gaussian exploration and then exploring action can be sampled from there
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)


env = gym.make('CartPole-v1')
actor = FunctionApproximation(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
critic = FunctionApproximation(input_size=env.observation_space.shape[0], output_size=env.action_space.n)


alpha = 0.001
gamma = 0.95
epsilon = 0.05
avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
agg_interval = 1
avg_reward = 0.0
avg_timestep = 0

# initialize policy and replay buffer
replay_buffer = ReplayBuffer()
start_episode = 0

# Play with a random policy and see
# run_current_policy(env.env, policy)

train_episodes = 200

# In[]:

# Train the network to predict actions for each of the states
for episode_i in range(start_episode, start_episode + train_episodes):
    episode_timestep = 0
    episode_reward = 0.0

    done = False

    cur_state = torch.Tensor(env.reset())

    while not done:
        # TODO : Use gaussian exploration for this
        action, log_prob = actor.select_action(cur_state)

        # take action in the environment
        next_state, reward, done, info = env.step(action)

        # compute the gradient from the sampled log probability
        loss = -log_prob * reward
        loss.backward()

        # add the transition to replay buffer
        # replay_buffer.add(cur_state, action, next_state, reward, done)

        # sample minibatch of transitions from the replay buffer
        # the sampling is done every timestep and not every episode
        # sample_transitions = replay_buffer.sample()

        # # update the policy using the sampled transitions
        # policy.update_policy(**sample_transitions)

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

# plot_timesteps_and_rewards(avg_history)
env.close()