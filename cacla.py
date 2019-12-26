import torch
import torch.nn as nn
import torch.optim as optim
import gym
from dqn import ReplayBuffer
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
import numpy as np


# TODO : 
#  1. Experience replay
#  2. Fixing target
#  3. Learning rate decay with a scheduler

class Critic(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=12):
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.output_layer(out)
        return out


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=12):
        super(Actor, self).__init__()
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


learning_rate = 0.001
train_episodes = 5000

env = gym.make('CartPole-v1')
actor = Actor(input_size=env.observation_space.shape[0], output_size=env.action_space.n)

# Approximating the Value function
critic = Critic(input_size=env.observation_space.shape[0], output_size=1)

optimizer_algo = 'adam'
if optimizer_algo == 'adam':
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
elif optimizer_algo == 'sgd':
    actor_optimizer = optim.SGD(actor.parameters(), lr=learning_rate, momentum=0.8)
    critic_optimizer = optim.SGD(critic.parameters(), lr=learning_rate, momentum=0.8)

gamma = 0.95
avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
agg_interval = 100
avg_reward = 0.0
avg_timestep = 0
running_loss1_mean = 0
running_loss2_mean = 0
loss1_history = []
loss2_history = []
# initialize policy and replay buffer
replay_buffer = ReplayBuffer()




# In[]:

# Train the network to predict actions for each of the states
for episode_i in range(train_episodes):
    episode_timestep = 0
    episode_reward = 0.0

    done = False

    cur_state = torch.Tensor(env.reset())


    # TODO : this has to be removed
    history = list()

    while not done:
        # TODO : Use gaussian exploration for this
        action, log_prob = actor.select_action(cur_state)

        # take action in the environment
        next_state, reward, done, info = env.step(action.item())

        # Update parameters of critic by TD(0)
        # TODO : Use TD Lambda here and compare the performance
        """
        u_value = critic(cur_state)
        target = reward + gamma * u_value
        critic_optimizer.zero_grad()
        loss1 = mse_loss(input=u_value, target=target)
        loss1.backward(retain_graph=True)
        running_loss1_mean += loss1.item()
        critic_optimizer.step()
        
        
        # Update parameters of actor by policy gradient
        actor_optimizer.zero_grad()
        # compute the gradient from the sampled log probability
        # TODO : Verify the computation here
        #  the log probability times the Q of the action that you just took in that state
        loss2 = -log_prob * (target - u_value) # the advantage function used is the TD error
        loss2.backward()
        running_loss2_mean += loss2.item()
        actor_optimizer.step()
        """

        # add the transition to replay buffer
        # replay_buffer.add(cur_state, action, next_state, reward, done)

        # sample minibatch of transitions from the replay buffer
        # the sampling is done every timestep and not every episode
        # sample_transitions = replay_buffer.sample()

        # # update the policy using the sampled transitions
        # policy.update_policy(**sample_transitions)

        episode_reward += reward
        episode_timestep += 1

        history.append([cur_state, next_state, action, log_prob, reward])

        cur_state = torch.Tensor(next_state)


    # TODO : This has to be removed
    #  Now calculate the return
    #  Here we calculate the return and update the loss in one step after the torch.sum function
    if optimizer_algo == 'adam':
        return_values = torch.Tensor()
        log_probabilities = torch.Tensor()
        for i in range(len(history)):
            return_t = 0
            el = 0
            for j in range(i, len(history)):
                return_t += np.power(gamma, el)*history[j][-1]
                el += 1
            return_values = torch.cat([return_values, torch.Tensor([return_t])])
            log_probabilities = torch.cat([log_probabilities, history[i][3].reshape(-1)])
        actor_optimizer.zero_grad()
        # -1 is important!!
        loss2 = torch.sum(torch.mul(-1*log_probabilities, return_values))
        loss2.backward()
        running_loss2_mean += loss2.item()
        actor_optimizer.step()

    elif optimizer_algo == 'sgd':
        for i in range(len(history)):
            return_t = 0
            el = 0
            for j in range(i, len(history)):
                return_t += np.power(gamma, el)*history[j][-1]
                el += 1
            actor_optimizer.zero_grad()
            # -1 is important!!
            loss2 = torch.sum(torch.mul(-1*history[i][3].reshape(-1), torch.Tensor([return_t])))
            loss2.backward()
            running_loss2_mean += loss2.item()
            actor_optimizer.step()

    loss1_history.append(running_loss1_mean/episode_timestep)
    loss2_history.append(running_loss2_mean/episode_timestep)
    running_loss1_mean = 0
    running_loss2_mean = 0

    avg_reward += episode_reward
    avg_timestep += episode_timestep

    avg_history['episodes'].append(episode_i + 1)
    avg_history['timesteps'].append(avg_timestep)
    avg_history['reward'].append(avg_reward)
    avg_timestep = 0
    avg_reward = 0.0

    if (episode_i + 1) % agg_interval == 0:
        print('Episode : ', episode_i+1, 'Loss : ', loss2_history[-1], 'Avg Timestep : ', avg_history['timesteps'][-1])

# In[]:
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
plt.subplots_adjust(wspace=0.5)
axes[0][0].plot(avg_history['episodes'], avg_history['timesteps'])
axes[0][0].set_title('Timesteps per episode')
axes[0][0].set_ylabel('Timesteps')
axes[0][1].plot(avg_history['episodes'], avg_history['reward'])
axes[0][1].set_title('Reward per episode')
axes[0][1].set_ylabel('Reward')
axes[1][0].set_title('Critic Loss')
axes[1][0].plot(loss1_history)
axes[1][1].set_title('Actor Loss')
axes[1][1].plot(loss2_history)

plt.show()


cur_state = env.reset()
total_step = 0
total_reward = 0.0
done = False
while not done:
    action, probs = actor.select_action(torch.Tensor(cur_state))
    next_state, reward, done, info = env.step(action.item())
    total_reward += reward
    env.render(mode='rgb_array')
    total_step += 1
    cur_state = next_state
print("Total timesteps = {}, total reward = {}".format(total_step, total_reward))
env.close()
