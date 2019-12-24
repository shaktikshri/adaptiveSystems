from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

# TODO : Variations possible
#  1. Use Replay Buffer
#  2. Freeze output for some iterations

# Define the policy and replay buffer
class FunctionApproximation:
    def __init__(self, env, lr, gamma, hidden_dim=24):
        self.state_dim = env.observation_space.shape[0]
        # TODO : need to check the dimension of the output, should it be 1?
        self.action_dim = env.action_space.n
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        self.q_model = Sequential()
        self.q_model.add(Dense(self.hidden_dim, input_dim=self.state_dim, activation='relu'))
        self.q_model.add(Dense(self.action_dim, activation='linear'))
        self.q_model.compile(loss='mse', optimizer=Adam(lr=self.lr))

    def update_policy(self, cur_states, actions, next_states, rewards, dones):
        # target doesnt change when its terminal, thus multiply with (1-done)
        # target = R(st-1, at-1) + gamma * max(a') Q(st, a')
        targets = rewards + np.multiply(1 - dones, self.gamma * (np.max(self.q_model.predict(next_states), axis=1)))

        # expanded_targets are the Q values of all the actions for the current_states sampled
        # from the previous experience. These are the predictions
        expanded_targets = self.q_model.predict(cur_states)

        # Prediction to be updated with the prediction+ground truth
        # We need to update the predictions to the values we want, which are the targets and then fit the model
        expanded_targets[list(range(len(cur_states))), actions] = targets

        self.q_model.fit(cur_states, expanded_targets, epochs=1, verbose=False)