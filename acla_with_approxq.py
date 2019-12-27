import torch
import torch.optim as optim
import gym
from dqn import ReplayBuffer
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from actor_critic_structure import Actor, Critic
from copy import deepcopy

# In[]:
# TODO :
#  2. Use Gaussian exploration while selecting the action
#  1. Use dropouts
#  2. LR Scheduler
#  3. Reward scaling in full batch mode


# TODO : Tweaking the learning rate
actor_learning_rate = 1e-2
critic_learning_rate = 1e-4
train_episodes = 5000

env = gym.make('CartPole-v1')
actor = Actor(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=24)

# Approximating the Value function
critic = Critic(input_size=env.observation_space.shape[0], output_size=1, hidden_size=24)
# critic_old is used for fixing the target in learning the V function
critic_old = deepcopy(critic)
copy_epoch = 100

optimizer_algo = 'sgd'

# Critic is always optimized in batch
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_learning_rate)

# actor is optimized either in batch or sgd
if optimizer_algo == 'sgd':
    actor_optimizer = optim.SGD(actor.parameters(), lr=actor_learning_rate, momentum=0.8, nesterov=True)
elif optimizer_algo == 'batch':
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_learning_rate)

# gamma = decaying factor
# TODO : Can change the step size
actor_scheduler = StepLR(actor_optimizer, step_size=500, gamma=0.1)
critic_scheduler = StepLR(critic_optimizer, step_size=200, gamma=0.1)


gamma = 0.99
avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
agg_interval = 10
avg_reward = 0.0
avg_timestep = 0
running_loss1_mean = 0
running_loss2_mean = 0
loss1_history = []
loss2_history = []
# initialize policy and replay buffer
replay_buffer = ReplayBuffer()

# In[]:


def update_critic(cur_states, actions, next_states, rewards, dones):
    # target doesnt change when its terminal, thus multiply with (1-done)
    # target = R(st-1, at-1) + gamma * max(a') Q(st, a')

    # move towards the fixed target, as per the older critic's prediction
    targets = rewards + torch.mul(1 - dones, gamma*critic(next_states).squeeze(-1) )

    # expanded_targets are the Q values of all the actions for the current_states sampled
    # from the previous experience. These are the predictions
    expanded_targets = critic(cur_states).squeeze(-1)
    critic_optimizer.zero_grad()
    loss1 = mse_loss(input=expanded_targets, target=targets)
    loss1.backward()
    critic_optimizer.step()
    return loss1.item()


# In[]:

# Train the network to predict actions for each of the states
for episode_i in range(train_episodes):

    # make a copy every copy_epoch epochs
    if episode_i % copy_epoch == 0:
        critic_old = deepcopy(critic)

    episode_timestep = 0
    episode_reward = 0.0

    done = False
    cur_state = torch.Tensor(env.reset())

    log_prob_list = torch.Tensor()
    u_value_list = torch.Tensor()
    target_list = torch.Tensor()

    while not done:
        # TODO : Use gaussian exploration for this
        action, log_prob = actor.select_action(cur_state)

        # take action in the environment
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.Tensor(next_state)

        u_value = critic(cur_state)
        # Update parameters of critic by TD(0)
        # TODO : Use TD Lambda here and compare the performance
        target = reward + gamma * critic(next_state)

        replay_buffer.add(cur_state, action, next_state, reward, done)
        # sample minibatch of transitions from the replay buffer
        # the sampling is done every timestep and not every episode
        sample_transitions = replay_buffer.sample_pytorch()
        # update the critic's q approximation using the sampled transitions
        running_loss1_mean += update_critic(**sample_transitions)

        if optimizer_algo == 'sgd':
            # Update parameters of actor by policy gradient
            actor_optimizer.zero_grad()
            # compute the gradient from the sampled log probability
            #  the log probability times the Q of the action that you just took in that state
            # TODO : the target here is still a moving target, see if fixing this for sometime leads to any improvement
            loss2 = -log_prob * (target - u_value) # the advantage function used is the TD error
            loss2.backward()
            running_loss2_mean += loss2.item()
            actor_optimizer.step()

        elif optimizer_algo == 'batch':
            target_list = torch.cat([target_list, target])
            u_value_list = torch.cat([u_value_list, u_value])
            log_prob_list = torch.cat([log_prob_list, log_prob.reshape(-1)])




        # TODO : This has to be removed, this form of training is not viable at all!
        if optimizer_algo == 'batch':
            # Update parameters of actor by policy gradient
            actor_optimizer.zero_grad()
            # compute the gradient from the sampled log probability
            #  the log probability times the Q of the action that you just took in that state
            loss2 = torch.sum(
                torch.mul(-log_prob_list, target_list - u_value_list))  # the advantage function used is the TD error
            loss2.backward()
            running_loss2_mean += loss2.item()
            actor_optimizer.step()




        episode_reward += reward
        episode_timestep += 1
        cur_state = next_state

    if optimizer_algo == 'batch':
        # Update parameters of actor by policy gradient
        actor_optimizer.zero_grad()
        # compute the gradient from the sampled log probability
        #  the log probability times the Q of the action that you just took in that state
        loss2 = torch.sum(torch.mul(-log_prob_list, target_list - u_value_list))  # the advantage function used is the TD error
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

    actor_scheduler.step()
    critic_scheduler.step()

    if (episode_i + 1) % agg_interval == 0:
        print('Episode : ', episode_i+1,
              'actor lr : ', actor_scheduler.get_lr(), 'critic lr : ', critic_scheduler.get_lr(),
              'Actor Loss : ', loss2_history[-1], 'Critic Loss', loss1_history[-1],
              'Avg Timestep : ', avg_history['timesteps'][-1])
        # print('Episode : ', episode_i+1,
        #     'critic lr : ', critic_scheduler.get_lr(),
        #     'Critic Loss', loss1_history[-1],
        #     'Avg Timestep : ', avg_history['timesteps'][-1])


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
axes[1][1].set_title('Actor Objective')
axes[1][1].plot(loss2_history)

plt.show()

# In[]:

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
