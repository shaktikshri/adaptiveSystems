import gym

env = gym.make('CartPole-v0')
# Full list of envs: https://gym.openai.com/envs

# put yourself in the start state
out = env.reset()
# out will contain the start states
# array([-0.04294398,  0.02886124,  0.02348346, -0.0215664 ])
# [Cartposition, cart velocity, pole angle, polevelocity at tip]

_ = env.action_space
# actions is a Discrete() object with 2 elements,
# 0: push cart to the left
# 1: push cart to the right

_ = env.observation_space
# Box(4,)

# observation, reward, done, info = env.step(action='Someactionarray')
# the observation, a reward, a done flag and an info dictionary for debugging
t = 0
done = False
while not done:
    t += 1
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action=action)
print(t)
env.close()

# In[]:

# Doing a random search for a dummy policy
# we'll chose actions randomly and see what happens
import numpy as np
import matplotlib.pyplot as plt

# a dummy policy to return 1 or 0, 1 is move to the left, 0 is move to the right
# as shown in the documentation for cartpole in openai/gym
def get_actions(s, w):
    if s.dot(w) > 0:
        return 1
    return 0


def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0

    while not done and t < 10000:
        # this shows a video as the game is being played
        # env.render()
        t += 1
        action = get_actions(observation, params)
        observation, reward, done, info = env.step(action)
        # note that we are ignoring the rewards here but they are all 1
    return t


def play_multiple_episode(env, T, params):
    episode_lengths = np.empty(T)
    # np.empty is faster since it doenst set all the values to 0 or 1

    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)
    avg = episode_lengths.mean()
    print('avg length of episodes : ', avg)
    return avg


def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    for t in range(100):
        print('Epoch : ', t)
        new_params = np.random.random(4)*2 -1
        avg_length = play_multiple_episode(env, 1000, new_params)
        episode_lengths.append(avg_length)

        # we want to pick the params which made the episode as long as possible
        # that means the game didnt end for a long time because the pole never fell down
        # for the entire episode
        if avg_length > best:
            best = avg_length
            params = new_params

    return episode_lengths, params

# In[]:

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()

    # now play the final episode with final weights
    # the weights are some parameters for the policy function
    print('Final one final episode with final weights')
    from gym import wrappers
    env = wrappers.Monitor(env, 'episode_shakti')
    play_one_episode(env, params)
    env.close()