# this is learning reward function from a sampled trajectory
import numpy as np
import matplotlib.pyplot as plt
from pulp import *
from tqdm.autonotebook import tqdm


# In[]:

# constructing a 5*5 gridworld as put by Andrew and Russel in Inverse Reinforcement Learning
# state is continuous [0,1]*[0,1]
class Agent:
    def __init__(self):
        self.cur_state = np.random.rand(2)
        # the action space is used for the policy selection in the deep q network
        self.action_space = self.obj()
    class obj:
        def __init__(self):
            pass
        def sample(self):
            return np.random.choice([0, 1, 2, 3])
    def step(self, action):
        assert action in [0, 1, 2, 3]
        # UP is 0, RIGHT 1, DOWN 2, LEFT 3
        if action == 0:
            step_size = np.array([0, 0.2])
        elif action == 1:
            step_size = np.array([0.2, 0])
        elif action == 2:
            step_size = np.array([0, -0.2])
        else:
            step_size = np.array([-0.2, 0])
        # whenever an action is performed, the guy moves in that direction and then a uniform noise from [-0.1, 0.1]
        # is added in each of the coordinates
        self.cur_state = self.cur_state + step_size + \
                         np.array([(2*np.random.rand() - 1)/10, (2*np.random.rand() - 1)/10])
        # Truncate the state to be in [0,1]*[0,1]
        self.cur_state[self.cur_state > 1] = 1
        self.cur_state[self.cur_state < 0] = 0
        return self.cur_state

"""
Remember whenever you're in lack of domain for basis functions, you can always take the domain to be the same
as the domain of the state space. And as always, Gaussian Mixtures are the best choice for basis functions
"""
# generate evenly spaced 15*15 2d gaussian over the state space
cov = [[0.1, 0],[0, 0.1]]
std = 0.1
mean = np.arange(0, 1, 1/15)
from scipy.stats import multivariate_normal

basis_functions = np.array([])
for i in range(15):
    for j in range(15):
        basis_functions = np.append(basis_functions, multivariate_normal(mean=[mean[i], mean[j]], cov=cov))
basis_functions = basis_functions.reshape(15, 15)
# TODO : Sanity check this, is this implementation correct ?
basis_functions = basis_functions.reshape(-1)

true_reward_function = lambda state: 1 if np.all(state >= np.array([0.8, 0.8])) else 0
# visualize the true reward distribution
# from plot_functions import figure
# x_points = np.arange(0, 1, 0.01)
# z = np.zeros((100, 100))
# for i in range(100):
#     for j in range(100):
#         z[i, j] = reward(np.array([x_points[i], x_points[j]]))
# figure(x_points, x_points, z, title='true reward')

env = Agent()

# In[]:
# We need a capability to find the optimal policy for any given reward distribution
"""This is where the algorithm is expensive, it needs you to find optimal policies for intermediate reward distributions
Now since the state space is continuous, any method– either discretization or Q learning will be expensive """
# Do a Q learning to learn the optimal policy for the reward structure given
# Using Q Learning to learn the optimal policy as per the reward distribution

from dqn import DQNPolicy, ReplayBuffer
def do_q_learning(env, reward_function, train_episodes, figure=False):
    alpha = 0.01
    gamma = 0.9
    epsilon = 0.1
    policy = DQNPolicy(env, lr=alpha, gamma=gamma, input=2, output=4)  # 4 actions output, up, right, down, left
    replay_buffer = ReplayBuffer()
    # Play with a random policy and see
    # run_current_policy(env.env, policy)
    agg_interval = 100
    avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
    # Train the network to predict actions for each of the states
    for episode_i in range(train_episodes):
        episode_timestep = 0
        episode_reward = 0.0
        env.__init__()
        # todo : the first current state should be 0
        cur_state = env.cur_state
        counter = 0
        done = False
        while not done:
            # Let each episode be of 30 steps
            counter += 1
            done = counter >= 30

            # todo : check if this line is working
            action = policy.select_action(cur_state.reshape(1, -1), epsilon)

            # take action in the environment
            next_state = env.step(action)
            reward = reward_function(next_state)

            # add the transition to replay buffer
            replay_buffer.add(cur_state, action, next_state, reward, done)

            # sample minibatch of transitions from the replay buffer
            # the sampling is done every timestep and not every episode
            sample_transitions = replay_buffer.sample()

            # update the policy using the sampled transitions
            policy.update_policy(**sample_transitions)

            episode_reward += reward
            episode_timestep += 1

            cur_state = next_state

        avg_history['episodes'].append(episode_i + 1)
        avg_history['timesteps'].append(episode_timestep)
        avg_history['reward'].append(episode_reward)

        learning_policy_progress.update()

    if figure:
        plt.plot(avg_history['episodes'], avg_history['reward'])
        plt.title('Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()
    return policy.q_model

# In[]:

# Get the true policy
true_policy = do_q_learning(env, true_reward_function, 500, figure=True)
# Now you have the true policy with you

# In[]:
# Now coming to the Linear Programming part. Here we will solve the
# optimization objective for the IRL from sampled trajectories

# first form the trajectories as per the given policy

all_policies = list()
gamma = 0.9
# form a random policy for now
# cur_policy = DQNPolicy(env, 0.01, 0.9, input=2, output=4)
cur_policy = true_policy
each_episode_length = 30
env = Agent()

N, un0, q = 30, 1, 0.9
u = np.empty((N,))
u[0] = un0
u[1:] = q
gamma_matrix = np.cumprod(u)

list_of_values_per_basis = np.empty((0, basis_functions.shape[0]))

# In[]:

def run_trajectories(cur_policy):
    env = Agent()
    values_all_trajectories = list()
    total_episodes = 5000
    for episode in range(total_episodes):
        done = False
        counter = 0
        env.cur_state = np.array([0,0]) # make sure the state is s0
        cur_state = env.cur_state
        trajectory = list()
        while not done:
            counter += 1
            done = counter >= each_episode_length
            # play the episode as per the latest policy
            trajectory.append(cur_state)
            # TODO : Sanity check this equation
            cur_state = env.step(cur_policy.predict(cur_state.reshape(1, -1))[0].argmax())
        # Now the trajectory is done, so find the value estimates for the different reward functions
        # trajectory is a 30*2 matrix, we'll apply the basis functions on it
        trajectory = np.array(trajectory)
        # values are all the 225 value functions as per different reward basis functions
        values = np.array([])
        for basis in basis_functions:
            values = np.append(values, np.dot(gamma_matrix.reshape(1, -1), basis.pdf(trajectory).reshape(-1, 1))[0][0])
        values_all_trajectories.append(values)
        trajectory_progress.update()
    values_all_trajectories = np.array(values_all_trajectories)
    # values_all_trajectories is a 5000*225 array
    values_per_basis = values_all_trajectories.mean(axis=0)
    return values_per_basis

# In[]:

true_values_per_basis = run_trajectories(true_policy) # it is the value of state(0,0) as per the best policy
# true_values_per_basis is a (225,) vector
policy = DQNPolicy(env, 0.01, 0.9, input=2, output=4).q_model

# In[]:

# Do the inductive step again and again
for iterations in range(1):
    print('Running Trajectory for the policy')
    trajectory_progress = tqdm(total=5000)
    list_of_values_per_basis = np.append(list_of_values_per_basis, run_trajectories(policy).reshape(1, -1), axis=0)
    # it is the value of state(0,0) as per the candidate policies
    # list_of_values_per_basis is a K*225 dimensional matrix where K is the number of candidate policies

    # Now need to do Linear Program
    prob = LpProblem('Sampled_Trajectory_Reward', LpMaximize)
    ALPHA = LpVariable.dicts('alpha', range(basis_functions.shape[0]))

    for i in ALPHA.keys():
        # the individual constraints on alphas
        ALPHA[i].lowBound = -1
        ALPHA[i].upBound = +1

    alphas = np.array([ALPHA[el] for el in range(basis_functions.shape[0])]).reshape(1, -1)
    # alphas is 1*225 dimensional vector, list_of_values_per_basis is a K*225 dimensional matrix
    v_s_star = np.dot(true_values_per_basis, alphas.T)
    v_s_pi = np.dot(list_of_values_per_basis, alphas.T)
    # v_s_pi is K*1 dimensional value of state s0 [s0=(0,0)] as per the candidate policies

    # TODO : Need to put this as a 2*prob, and define the probability piecewise
    prob += lpSum(v_s_star - v_s_pi)

    status = prob.solve()
    if status != 1:
        raise Exception('Optimal alpha not found')
    # Now you'll get the setting of the individual alphas
    found_alphas = np.array([value(ALPHA[el]) for el in range(basis_functions.shape[0])])

    # Now you need to find the reward function
    found_reward = lambda state: np.sum([found_alphas[el]*basis_functions[el].pdf(state) for el in range(basis_functions.shape[0])])

    # Then get the policy as per that reward function,
    print('Finding the optimal policy now')
    learning_policy_progress = tqdm(total=500)
    policy = do_q_learning(env, found_reward, 500)
    # todo : can change the number of episodes here for
    #  learning the policy

    # add the policy to the list of current policies, add its value function to the list
    # this is happening in the next iteration of the loop
    # Repeat
    print('Will now run trajectory for the found policy, going in the next iteration')

    # visualize the found reward distribution
    from plot_functions import figure
    x_points = np.linspace(0, 1, 50)
    z = np.zeros((50, 50))
    for i in range(50):
        print('In iteration ',i)
        for j in range(50):
            z[i, j] = found_reward(np.array([x_points[i], x_points[j]]))
    figure(x_points, x_points, z, title='Found reward')
