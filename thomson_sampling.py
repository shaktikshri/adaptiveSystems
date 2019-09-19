#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:02:12 2019

@author: shaktikumar
"""


import random
import pandas as pd
import matplotlib.pyplot as plt


# the data input will be from Jerrod's simulation
def gather_data_from_sim():
    # Sim to throw the data
    data = {}
    data['B1'] = [random.randint(1, 5) for x in range(200)]
    data['B2'] = [random.randint(0, 1) for x in range(200)]
    data['B3'] = [random.randint(0, 1) for x in range(200)]
    data['B4'] = [random.randint(0, 1) for x in range(200)]
    data['B5'] = [random.randint(0, 1) for x in range(200)]
    return data


data = pd.DataFrame(gather_data_from_sim())
observations = 200
machines = 5

machine_selected = []
rewards = [0] * machines
penalties = [0] * machines
total_reward = 0
for n in range(0, observations):
    bandit = 0
    beta_max = 0
    for i in range(0, machines):
        beta_d = random.betavariate(rewards[i] + 1, penalties[i] + 1)
        if beta_d > beta_max:
            beta_max = beta_d
            bandit = i
    machine_selected.append(bandit)
    reward = data.values[n, bandit]
    if reward >= 1:
        rewards[bandit] = rewards[bandit] + 1
    else:
        penalties[bandit] = penalties[bandit] + 1
    total_reward = total_reward + reward

print("\n\nRewards By Machine = ", rewards)
print("\nTotal Rewards = ", total_reward)
print("\nMachine Selected At Each Round By Thompson Sampling : \n", machine_selected)

# Visualizing the rewards of each machine
plt.bar(['B1','B2','B3','B4','B5'],rewards)
plt.title('MABP')
plt.xlabel('Bandits')
plt.ylabel('Rewards By Each Machine')
plt.show()

# Number Of Times Each Machine Was Selected
from collections import Counter
print("\n\nNumber Of Times Each Machine Was Selected By The Thompson Sampling Algorithm : \n",dict(Counter(machine_selected)))

# Visualizing the Number Of Times Each Machine Was Selected
plt.figure()
plt.hist(machine_selected)
plt.title('Histogram of machines selected')
plt.xlabel('Bandits')
plt.xticks(range(0, 5))
plt.ylabel('No. Of Times Each Bandit Was Selected')
plt.show()