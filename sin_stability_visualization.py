import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


def f(x, mean=0.0, std=0.0):
    noise = np.random.normal(loc=mean, scale=std, size=x.shape[0])
    return np.sin(x) + noise

std = 0.05
mean = 0
total_size = 10

total_range = np.arange(0, total_size, 0.1)
for std in [0, 0.2, 0.4, 0.8, 1]:
    plt.figure()
    plt.plot(f(total_range, mean, std))
# plt.xlim(100, 200)