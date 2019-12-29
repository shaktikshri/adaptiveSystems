### Architecture

1. Critic : Network to approximate the value function <br> state_dimension -> 12 -> 12 -> 12 -> 1
2. Actor : Network with state_dimension -> 12 -> 12 -> 12 -> n_actions

Critic uses TD(0) learning + experience replay with buffer size 32. 
<br>Actor uses 1 step TD error + full batch gradient descent +  error normalization.
<br>Learning rates are 0.01 for both networks.
<br>Additional reward scaling was done to speeden the convergence. Note that this scaling can be removed and the results remain the same. 

### Training
```
Episode :  10 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  -0.010201778676774766 Critic Loss 588.0338168674045 Avg Timestep :  18
Episode :  20 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  0.006513764983729312 Critic Loss 477.68187994706005 Avg Timestep :  19
Episode :  30 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  -0.1845457323135868 Critic Loss 388.1130636892011 Avg Timestep :  31
Episode :  40 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  0.009060933401710108 Critic Loss 349.33526310167815 Avg Timestep :  76
Episode :  50 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  0.006917795382047954 Critic Loss 374.57152581465874 Avg Timestep :  95
Episode :  60 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  -0.13249082918520327 Critic Loss 338.935127682156 Avg Timestep :  108
Episode :  70 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  -0.035215601921081544 Critic Loss 518.0776229858399 Avg Timestep :  200
Episode :  80 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  -0.05176480770111084 Critic Loss 455.6855531311035 Avg Timestep :  200
Episode :  90 actor lr :  [0.01] critic lr :  [0.01] Actor Objective :  -0.04757546901702881 Critic Loss 515.156781463623 Avg Timestep :  200
```

![Experience Replay Critic + Batch Update Actor](RL_Benchmarks/fig2.png)

### Testing
```
Episode : 1 : Total timesteps = 200, total reward = 200.0
Episode : 2 : Total timesteps = 200, total reward = 200.0
Episode : 3 : Total timesteps = 200, total reward = 200.0
Episode : 4 : Total timesteps = 200, total reward = 200.0
Episode : 5 : Total timesteps = 200, total reward = 200.0
```