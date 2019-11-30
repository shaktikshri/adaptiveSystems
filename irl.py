# Implementation of Finite State IRL as Andrew and Russell, https://ai.stanford.edu/~ang/papers/icml00-irl.pdf
# The cleaning robot's Reward function is recovered

import numpy as np

T_s_a_sbar = np.zeros(shape=(25, 4, 25))

# 5*5 gridworld same as the cleaning robot. With start at (5,5) and end at (4,4)
# 0.8 chance of moving in the correct direction, 0.1,0.1 chance of moving in the direction perpendicular to it
# If the bot hits the wall while moving, it says where it is
for row in range(5):
    for col in range(5):
        state_index = row*5 + col

        possible_next_states = [min(row+1, 4)*5 + col, # state towards up
                                row*5 + min(col+1, 4),  # state towards right
                                max(row-1,0)*5 + col, # state towards down
                                row*5 + max(col-1,0)] # state towards left
        for a in range(4): # action 0: Up, 1: Right, 2: Down, 3: Left
            prob = np.zeros((4))
            prob[a] = 0.8
            prob[(a - 1) % 4] = 0.1
            prob[(a + 1) % 4] = 0.1

            # visualize this with action UP
            T_s_a_sbar[state_index, a, possible_next_states[0]] += prob[0]     # state towards UP gets 0.8
            T_s_a_sbar[state_index, a, possible_next_states[1]] += prob[1]     # state to the right gets 0.1
            T_s_a_sbar[state_index, a, possible_next_states[2]] += prob[2]       # state to the down gets 0
            T_s_a_sbar[state_index, a, possible_next_states[3]] += prob[3]     # state to the left gets 0.1
