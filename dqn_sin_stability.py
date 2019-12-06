import numpy as np
x = np.arange(0,2*np.pi,0.01)

def f(x, noise):
    return np.sin(x) + noise

# f = lambda x: np.sin(x)
import matplotlib.pyplot as plt
noise = np.random.choice([0.025, -0.025, 0.05, -0.05], size=x.shape[0])
states = np.array([x, f(x, noise)])
plt.plot(states[0], states[1], label='original', color='b')

def step(states, actions):
    states[1] += actions
    return states

actions = -noise
states = step(states, actions)
plt.plot(states[0], states[1], label='smoothened', color='g')
plt.legend()
plt.show()

# In[]:

class RandomVariable():
    def __init__(self, errepsilon):
        self.y = 0
        self.x = 0
        # TODO : Change these values and check
        self.errepsilon = errepsilon

    def step(self, action):
        self.y += action
        reward = self.get_reward()
        self.x += 1
        return np.array([self.x, self.y]), reward

    def f(self):
        return np.sin(self.x)

    def get_reward(self):
        if np.abs(self.f() - self.y) < self.errepsilon:
            # TODO : Change the reward values and check
            return +10
        else:
            return -1

# In[]: