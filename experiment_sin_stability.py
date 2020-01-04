import torch
import torch.optim as optim
from dqn import ReplayBuffer
# from torch.distributions import Categorical
from torch.nn.functional import mse_loss
import numpy as np
# from torch.optim.lr_scheduler import StepLR
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from actor_critic_structure import Actor, Critic
from copy import deepcopy
from env_definition import RandomVariable
# import json

# In[]:

class ActorReplayBuffer:
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.target = []
        self.predicted = []
        self.gradient = []

    def __len__(self):
        return len(self.target)

    def add(self, target, predicted, gradient):
        self.target.append(target)
        self.predicted.append(predicted)
        self.gradient.append(gradient)

    def sample(self, sample_size=32):
        sample_objectives = {}
        if self.__len__() >= sample_size:
            # pick up only random 32 events from the memory
            indices = np.random.choice(self.__len__(), size=sample_size)
            sample_objectives['target'] = torch.stack(self.target)[indices].squeeze(-1)
            sample_objectives['predicted'] = torch.stack(self.predicted)[indices].squeeze(-1)
            sample_objectives['gradient'] = torch.stack(self.gradient)[indices].squeeze(-1)
        else:
            # if the current buffer size is not greater than 32 then pick up the entire memory
            sample_objectives['target'] = torch.stack(self.target).squeeze(-1)
            sample_objectives['predicted'] = torch.stack(self.predicted).squeeze(-1)
            sample_objectives['gradient'] = torch.stack(self.gradient).squeeze(-1)

        return sample_objectives

# In[]:

actor_learning_rate = 1e-3
critic_learning_rate = 1e-3
train_episodes = 3000

highest = 10
intermediate = 1
penalty = -1
lowest = -10

env = RandomVariable(highest=highest, intermediate=intermediate, penalty=penalty, lowest=lowest)
# The actor can just output an action, since the action space is continuous now
actor = Actor(input_size=env.observation_space.shape[0], output_size=1, hidden_size=24, continuous=True)

# Approximating the Value function
critic = Critic(input_size=env.observation_space.shape[0], output_size=1, hidden_size=24)

# critic_old is used for fixing the target in learning the V function
critic_old = deepcopy(critic)

copy_epoch = 100

optimizer_algo = 'batch'

# Critic is always optimized in batch
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_learning_rate)

if optimizer_algo == 'sgd':
    actor_optimizer = optim.SGD(actor.parameters(), lr=actor_learning_rate, momentum=0.8, nesterov=True)
elif optimizer_algo == 'batch':
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_learning_rate)

# actor_scheduler = StepLR(actor_optimizer, step_size=500, gamma=1)
# critic_scheduler = StepLR(critic_optimizer, step_size=500, gamma=1)

gamma = 1 # gamma = 0 because the action you took in the current state is no way going to affect what you have to do
# in the next state
avg_history = {'episodes': [], 'timesteps':[], 'reward': [], 'hits percentage' : []}
agg_interval = 10
running_loss1_mean = 0
running_loss2_mean = 0
loss1_history = []
loss2_history = []
# initialize policy and replay buffer
replay_buffer = ReplayBuffer()
actor_replay_buffer = ActorReplayBuffer()

beta = 0.001  # beta is the momentum in variance updates of TD Error
running_variance = 1


# In[]:


def update_critic(critic_old, cur_states, actions, next_states, rewards, dones):

    # target doesnt change when its terminal, thus multiply with (1-done)
    targets = rewards + torch.mul(1 - dones, gamma*critic_old(next_states).squeeze(-1) )
    # expanded_targets are the Q values of all the actions for the current_states sampled
    # from the previous experience. These are the predictions
    expanded_targets = critic(cur_states).squeeze(-1)
    critic_optimizer.zero_grad()
    # detach the targets from the computation graph
    loss1 = mse_loss(input=targets.detach(), target=expanded_targets)  # the implementation is (input-target)^2
    loss1.backward()
    critic_optimizer.step()
    return loss1.item()


# In[]:
# Train the network to predict actions for each of the states
for episode_i in range(train_episodes):

    # make a copy every copy_epoch epochs
    if episode_i % copy_epoch == 0:
        critic_old = deepcopy(critic)

    episode_reward = 0.0
    episode_timestep = 0

    done = False
    cur_state = torch.Tensor(env.reset())

    actors_output_list = torch.Tensor()
    action_target_list = torch.Tensor()
    u_value_list = torch.Tensor()
    target_list = torch.Tensor()
    reward_list = list()

    while not done:
        action, _ = actor.select_action(cur_state)

        # take action in the environment
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.Tensor(next_state)
        reward_list.append(reward)

        u_value = critic(cur_state)
        u_value_list = torch.cat([u_value_list, u_value])

        # Update parameters of critic by TD(0)
        target = reward + gamma * (1-done) * critic_old(next_state)
        target_list = torch.cat([target_list, target])

        # replay_buffer.add(cur_state, action, next_state, reward, done)
        # sample_transitions = replay_buffer.sample_pytorch(sample_size=32)
        # # update the critic's q approximation using the sampled transitions
        # running_loss1_mean += update_critic(critic_old, **sample_transitions)

        if target - u_value > 0:
            if optimizer_algo == 'sgd':
                # Update parameters of actor by ACLA
                td_error = target - u_value
                running_variance = running_variance*(1-beta) + beta*torch.pow(td_error, 2)
                # no. of updates to this action should be equal to floor(TD Error / std_dev of TD error) as per the
                # original paper in Hasselt and Wiering
                for el in range(int(torch.ceil(td_error / torch.sqrt(running_variance)))):
                    actor_optimizer.zero_grad()
                    loss2 = mse_loss(input=action.detach(), target=actor(cur_state)) # the implementation for mse
                    # is (input - target)^2
                    loss2.backward()
                    actor_optimizer.step()
                    running_loss2_mean += loss2.item()

            elif optimizer_algo == 'batch':
                action_target_list = torch.cat([action_target_list, action])
                actors_output_list = torch.cat([actors_output_list, actor(cur_state)])
                # log_prob_list = torch.cat([log_prob_list, log_prob.reshape(-1)])

        episode_reward += reward
        episode_timestep += 1
        cur_state = next_state

    critic_optimizer.zero_grad()
    u_value_list_copy = (u_value_list - u_value_list.mean()) / u_value_list.std()
    target_list_copy = (target_list - target_list.mean()) / target_list.std()
    loss1 = mse_loss(input=u_value_list_copy, target=target_list_copy.detach())
    loss1.backward(retain_graph=True)
    running_loss1_mean += loss1.item()
    critic_optimizer.step()

    # Do the loss backward only if there was at least 2 transitions in the episode with TD error > 0
    # there wont be any elements in action_target_list, action_list of the episode has no TD error > 0
    if action_target_list.shape[0] >= 2:
        if optimizer_algo == 'batch':
            # Update parameters of actor by policy gradient
            actor_optimizer.zero_grad()
            # TODO : The updates should be of size proportional to the variance reduction
            loss2 = mse_loss(input=action_target_list, target=actors_output_list)
            loss2.backward()
            running_loss2_mean += loss2.item()
            actor_optimizer.step()

    loss1_history.append(running_loss1_mean/episode_timestep)
    loss2_history.append(running_loss2_mean/episode_timestep)
    running_loss1_mean = 0
    running_loss2_mean = 0

    avg_history['timesteps'].append(episode_timestep)
    avg_history['reward'].append(episode_reward)
    avg_history['hits percentage'].append((reward_list.count(intermediate) + reward_list.count(highest))/len(reward_list))

    # actor_scheduler.step()
    # critic_scheduler.step()

    if (episode_i + 1) % agg_interval == 0:
        print(
              'Episode : ', episode_i+1,
                # 'actor lr : ', actor_scheduler.get_lr(), 'critic lr : ', critic_scheduler.get_lr(),
                'Actor Loss : ', loss2_history[-1], 'Critic Loss : ', loss1_history[-1],
                #  'Timestep : ', avg_history['timesteps'][-1], 'Reward : ',avg_history['reward'][-1])
                'Hits : ', avg_history['hits percentage'][-1],
                'Timestep : ', avg_history['timesteps'][-1]
        )

# In[]:

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 7))
plt.subplots_adjust(wspace=0.5)
axes[0].plot(avg_history['hits percentage'])
axes[0].set_title('Hits+Partial Hits per episode')
axes[0].set_ylabel('Hits')
axes[1].set_title('Critic Loss')
axes[1].plot(loss1_history)
axes[2].set_title('Actor Objective')
axes[2].plot(loss2_history)

# In[]:

cur_state = env.reset()
total_step = 0
total_reward = 0.0
done = False
y1 = list()
y2 = list()
x = list()
while not done:
    x.append(cur_state[0])
    action, probs = actor.select_action(torch.Tensor(cur_state))
    y1.append(cur_state[1]+action.item())
    y2.append(cur_state[1])
    next_state, reward, done, info = env.step(action.item())
    total_reward += reward
    total_step += 1
    cur_state = next_state
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
plt.subplots_adjust(wspace=0.5)
axes[0].scatter(x, y2)
axes[0].set_title('Original')
axes[1].scatter(x, y1)
axes[1].set_title('Corrected')

plt.show()
