# this is learning reward function from a sampled trajectory, which will be the case in our
# interaction with engineers during demonstration by expert

from env_definition import RandomVariable
import numpy as np

env = RandomVariable(highest=10, intermediate=5, lowest=-5, penalty=-10)
policy = list()
for episodes in range(100):
    done = False
    cur_state = env.reset()
    while not done:
        # we need to construct the optimal policy, so need to append the state action pair for all
        # possible states here
        action = -1*[0.1, 0.5, -0.3, -0.5][int(env.time // 2)]
        policy.append([cur_state, action])
        next_state, _, done, _ = env.step(action)
        cur_state = next_state

import matplotlib.pyplot as plt
plt.scatter([el[0][0] for el in policy], [el[0][1]+el[1] for el in policy])
