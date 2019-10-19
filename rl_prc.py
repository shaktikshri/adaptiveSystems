import numpy as np

T = np.array([
    [0.9, 0.1],
    [0.5, 0.5]
])

for i in [1,2,3,4,5]:
    T_i = np.linalg.matrix_power(T, i)
    print('Transition Prob after time t=',i,'\n',T_i)
# the transition probability converges to array([
#           [0.83333333, 0.16666667],
#           [0.83333333, 0.16666667]
#     ])
# Thus the asymptotic belief is that after infinite steps, the state will be
# system will remain in s0 with a prob of 83.3% and move to s1 16.7% of the time
# Upon moving to s1, it will stay in s1 16.7% of the time, and move to s0
# with 83.3% probability
