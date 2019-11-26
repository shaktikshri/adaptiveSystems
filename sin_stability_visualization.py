import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


def f(x, mean=0.0, std=0.0):
    noise = np.random.normal(loc=mean, scale=std, size=x.shape[0])
    noise = np.random.choice([0.25,-0.25,0.5,-0.5], size=x.shape[0])
    return np.sin(x) + noise

std = 0.05
mean = 0
total_size = 10

total_range = np.arange(0, total_size, 0.1)
# Plotting 10 times the sin function with a std dev of 0.1 in the gaussian noise
# gives the platform for stabilizing the function
plt.figure()
plt.plot(f(total_range, mean, 0.1), color='g')