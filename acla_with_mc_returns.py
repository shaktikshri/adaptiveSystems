import torch
import torch.optim as optim
import gym
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from actor_critic_structure import Actor, Critic

# In[]:

learning_rate = 1e-4
train_episodes = 5000

env = gym.make('CartPole-v1')
actor = Actor(input_size=env.observation_space.shape[0], output_size=env.action_space.n)

# Approximating the Value function
critic = Critic(input_size=env.observation_space.shape[0], output_size=1)

optimizer_algo = 'sgd'
decaying_algo = 'step'

if optimizer_algo == 'adam':
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
elif optimizer_algo == 'sgd':
    actor_optimizer = optim.SGD(actor.parameters(), lr=learning_rate, momentum=0.8, nesterov=True)
    critic_optimizer = optim.SGD(critic.parameters(), lr=learning_rate, momentum=0.8, nesterov=True)

    if decaying_algo == 'step':
        # gamma = decaying factor
        scheduler = StepLR(actor_optimizer, step_size=1000, gamma=0.1)

    # We cant use plateau decay here since the gradient is very noisy for stochastic estimates,
    # nothing plateaus at all !
    # elif decaying_algo == 'plateau':
    #     # patience: number of epochs - 1 where loss plateaus before decreasing LR
    #     # patience = 0, after 1 bad epoch, reduce LR. New lr = lr * factor
    #     scheduler = ReduceLROnPlateau(actor_optimizer, mode='max', factor=0.1, patience=0, verbose=True)

gamma = 0.99
avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
agg_interval = 100
avg_reward = 0.0
avg_timestep = 0
running_loss1_mean = 0
running_loss2_mean = 0
loss1_history = []
loss2_history = []


# In[]:

# Train the network to predict actions for each of the states
for episode_i in range(train_episodes):
    episode_timestep = 0
    episode_reward = 0.0

    done = False
    cur_state = torch.Tensor(env.reset())

    if optimizer_algo == 'sgd':
        scheduler.step()

    history = list()

    while not done:
        action, log_prob = actor.select_action(cur_state)
        # take action in the environment
        next_state, reward, done, info = env.step(action.item())
        episode_reward += reward
        episode_timestep += 1
        history.append([cur_state, next_state, action, log_prob, reward])
        cur_state = torch.Tensor(next_state)

    #  Now calculate the return
    if optimizer_algo == 'adam':
        # calculate the expected return and update the parameter wrt the expected gradient of the objective function
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
        # Scale rewards to reduce variance
        return_values = (return_values - return_values.mean()) / return_values.std()
        # -1 is important!!
        loss2 = torch.sum(torch.mul(-1*log_probabilities, return_values))
        loss2.backward()
        running_loss2_mean += loss2.item()
        actor_optimizer.step()

    elif optimizer_algo == 'sgd':
        # calculate the return for each transition and update the parameter wrt
        # stochastic gradient of the objective function
        for i in range(len(history)):
            return_t = 0
            el = 0
            for j in range(i, len(history)):
                return_t += np.power(gamma, el)*history[j][-1]
                el += 1
            actor_optimizer.zero_grad()
            # here reward scaling cannot be done since no batch is available to us at all
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
        print('Episode : ', episode_i+1, 'Learning Rate', scheduler.get_lr(), 'Loss : ', loss2_history[-1], 'Avg Timestep : ', avg_history['timesteps'][-1])

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
