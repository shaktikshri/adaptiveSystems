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
plt.xlim(0,100)

