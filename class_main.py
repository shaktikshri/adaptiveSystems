#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:02:12 2019

@author: shaktikumar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from IPython.display import HTML, display
import tabulate


class AdaptiveLearning:
    def __init__(self, versions=None, prior=None):  # Taking in version set and prior as input
        self.trials = np.zeros((len(versions),), dtype=int)
        self.successes = np.zeros_like(self.trials)
        self.versions = versions
        if prior is None:
            self.prior = [(1.0, 1.0) for i in range(len(versions))]

    def add_data(self, version_num, success):
        self.trials[version_num] += 1
        if success:
            self.successes[version_num] += 1

    def get_distr(self):
        posterior_sample = np.zeros(len(self.versions))
        x = []
        params = []
        for i in range(len(self.versions)):
            a = prior[i][0] + self.successes[i]  # alpha
            b = prior[i][1] + self.trials[i] - self.successes[i]  # beta; (trials - successes) are the failures
            params.append([a, b])  # appending these parameters for plotting graphs
            x += [np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)]
            posterior_sample[i] = np.random.beta(a, b)  # choosing a random sample from the beta distr

        probabilities = self.get_relative_probabilities()  # retrieve the probabilities using the no. of times
        # a version has been selected
        return np.argmax(posterior_sample), x, params, probabilities
        # returning the maximum sample's version_id, alongwith a few plotting stuff

    def get_relative_probabilities(self, iterations=100):
        times_selected = np.zeros(len(self.versions))
        for i in range(iterations):
            max_beta = 0
            version_max_beta = None
            for k in range(len(self.versions)):
                a = prior[k][0] + self.successes[k]  # alpha
                b = prior[k][1] + self.trials[k] - self.successes[k]  # beta
                curr_beta = np.random.beta(a, b)
                if curr_beta > max_beta:
                    max_beta = curr_beta
                    version_max_beta = k
            times_selected[version_max_beta] += 1
        probabilities = [float(times_selected[version]) / sum(times_selected) for version in range(len(times_selected))]
        return probabilities


def sample_beta(success_proportion, num_trials):
    return [round(num_trials * success_proportion), round(num_trials * (1 - success_proportion))]



versions = ['survey', 'question', 'returning']
realpriors = [[15, 5], [1, 1], [1, 1]]
somedata = [[0, 0], [5, 5], [0, 0]]
prior = np.add(realpriors, somedata)

num_learners = 1000  # change the number of observations here

posterior_reward_rate = {'survey': 0.5, 'question': 0.5, 'returning': 0.5}  # change the average reward here (0 - 1)
# this is basically the posterior prob dist

probability_iterations = 1000  # Just for computing probability of condition accurately

# ----------------------------------------------------------------------------------

e = AdaptiveLearning(versions, prior)
probabilities = e.get_relative_probabilities(probability_iterations)
# --------------
# display the probability table
table = zip(versions, probabilities)
display(HTML(tabulate.tabulate(table, tablefmt='html')))

plt.figure(figsize=(20, 6))
# Plotting the initial prior graphs
for i in range(len(versions)):
    plt.subplot(3, 2, 2 * i + 1)
    x = np.linspace(beta.ppf(0.01, prior[i][0], prior[i][1]), beta.ppf(0.99, prior[i][0], prior[i][1]), 100)
    plt.plot(x, beta.pdf(x, prior[i][0], prior[i][1]), label='beta pdf')
    plt.title(versions[i] + '- prior')
    plt.xlim([0, 1])
plt.show()
prior_init = prior



# import random
# df = pd.read_csv('email_data.csv', usecols = ['ndaysactUSER','agequartilesUSER','subjectCondition','started_survey'])
df = pd.DataFrame(columns=["subjectCondition", "started_survey"])
for i in range(num_learners):
    version = np.random.choice(versions)
    rand = np.random.random()
    started_survey = 0
    if rand <= posterior_reward_rate[version]:
        started_survey = 1
    df = df.append({"subjectCondition": version, "started_survey": started_survey}, ignore_index=True)
# df = pd.DataFrame(data = {"subjectCondition": random.choices(versions, k=num_learners), "started_survey": np.random.randint(2, size=num_learners)})
print(df.head())


# Getting succes information from the data
vers_dict = {'survey': 0, 'question': 1, 'returning': 2}
df = df.replace({'subjectCondition': vers_dict})
print(
    'Success Rate of "Survey":{0}/{1} '.format(df['started_survey'].loc[df['subjectCondition'] == 0].tolist().count(1),
                                               df['subjectCondition'].tolist().count(0)))
print('Success Rate of "Question":{0}/{1} '.format(
    df['started_survey'].loc[df['subjectCondition'] == 1].tolist().count(1), df['subjectCondition'].tolist().count(1)))
print('Success Rate of "Returning":{0}/{1} '.format(
    df['started_survey'].loc[df['subjectCondition'] == 2].tolist().count(1), df['subjectCondition'].tolist().count(2)))

"""#Graphing the Posterior Distribution"""

# prior = prior_init
# tried_outputs = [0,0,0]
# params = []

scores = [0, 0, 0]

# Now creating the posterior distribution as data is added.
# 1. Iterates over all the df indices,
# 2. updates the data in email_mooclet object,
# 3. and finds the posterior prob. of each along with the the version with the highest probability
for trial in range(len(df.index)):
    e = AdaptiveLearning(versions, prior)
    # our initial prior = real_prior + some_noise
    e.add_data(df.loc[trial, 'subjectCondition'], df.loc[trial, 'started_survey'])
    result, x, prior, probabilities = e.get_distr()
    # result is the index of the maximum probability version
    scores[result] += 1
    # increment that version by 1 saying it succeeded

plt.figure(figsize=(20, 6))
# print(probabilities)
# Checking how many times each version won the sampling contest
table = zip(versions, probabilities)
display(HTML(tabulate.tabulate(table, tablefmt='html')))
for i in range(len(versions)):  # As one particular version gets chosen more, it's probability of
    # print(x)
    plt.subplot(3, 2, (i + 1) * 2)  # winning should increase. Here versions chosen at random
    plt.plot(x[i], beta.pdf(x[i], prior[i][0], prior[i][1]), label='beta pdf')
    plt.title(versions[i])
    plt.xlim([0, 1])
    betas = beta.pdf(x[i], prior[i][0], prior[i][1])
    # print(betas)
    max_y = max(betas)
    max_x = x[i][max_y.argmax()]
    plt.text(max_x, max_y, str((max_x)))  # Plotting final posterior graph
plt.tight_layout()
plt.show()

