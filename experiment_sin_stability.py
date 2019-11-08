import numpy as np
import matplotlib.pyplot as plt


def f(x, mean, std):
    return np.sin(x) + np.random.normal(loc=mean, scale=std)


total_size = 1000
a = list()
for el in range(total_size):
    a.append(f(el/10, 0, 0.5))
a = np.array(a)
plt.plot(np.arange(1,total_size+1), a)
plt.xlim(100, 200)


class RandomVariable:
    def __init__(self):
        self.mean = 0
        self.std = 0.5
        self.state = 0
        self.curr_val = 0
        self.counter = 0
        self.safe_mod = 3
        self.unsafe_mod = 5

    def get_state(self):
        # defined 3 states, safe (0), unsafe (1), critical (2)
        # Safe : value hops between [-3, +3]
        # unsafe : value between [-5,-3) and (+3, +5]
        # critical : values beyond that
        if abs(self.curr_val) < self.safe_mod:
            return 0
        elif abs(self.curr_val) < self.unsafe_mod:
            return 1
        return 2

    def f(self, mean, std):
        return np.sin(self.curr_val) + np.random.normal(loc=mean, scale=std)

    def get_value(self):
        self.counter += 1
        return self.f(self.mean, self.std)