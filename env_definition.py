import numpy as np

class RandomVariable:
    def __init__(self):
        self.mean = 0
        self.std = 0.5
        self.state = 0
        self.curr_val = 0
        self.safe_mod = 0.01
        self.unsafe_mod = 0.1
        self.critical_mod = None
        self.reward_matrix = None
        self.action_to_value_mapping = None # this is the action to value mapping,
        # i.e. for a given action how much should you add/subtract from the current_value
        self.noise_array = [-0.5, -0.25, 0.25, 0.5] # this noise array should be used as a
        # PoC to show that with 4 distinct values of noises we can segregate the noise.
        # Once we've shown this, we can extend this to large number of noise values, maybe
        # even continuous

    def set_reward_matrix(self, matrix):
        self.reward_matrix = matrix

    def set_action_to_value_mapping(self, matrix):
        self.action_to_value_mapping = matrix

    def reset(self, exploring_starts):
        if exploring_starts:
            # Randomly select a current value
            low, high = -self.critical_mod-2, self.critical_mod+2
        else:
            low, high = -self.safe_mod-2, self.safe_mod+2
        self.curr_val = np.random.uniform(low=low, high=high)
        # Set the state according to the metric
        self.evaluate_state()
        return self.state, self.curr_val

    def evaluate_state(self, action_value, t):
        """
        defined 5 states,
        (State 2)       Safe : value hops between [-1, +1] for the taken sin function
        (State 1 and 3) unsafe : value between [-1.01,-1) and (+1, +1.01]
        (State 0 and 4) critical : values between [-inf, -1.01) and (+1.01, +inf]
        Out of these only 'critical' are the terminal states
        the episodes will end at it (meaning there is nothing you can do
        to stabilize the value in the critical state, and you failed)
        :return:
        """
        difference = self.f(t) - self.f(self.curr_val+action_value)
        if abs(difference) < self.safe_mod:
            self.state = 2
        elif difference < self.unsafe_mod:
            # difference is positive, thus this is the case of underestimation
            self.state = 3
        elif difference > -self.unsafe_mod:
            # difference is negative, thus this is the case of overestimation
            self.state = 1
        else:
            # else you've failed!
            self.state = 0 if difference > 0 else 4

    def f(self, x, mean=0.0, std=0.0):
        # return np.sin(x) + np.random.normal(loc=mean, scale=std)
        return np.sin(x) + np.random.choice(self.noise_array)

    def get_value(self, mean=None, std=None):
        if not mean and not std:
            return self.f(self.curr_val, self.mean, self.std)
        else:
            return self.f(self.curr_val, mean, std)

    def step(self, action):
        """
        Takes the given action and returns the new state,
        the new curr_val, the reward and a flag to show if the new state
        is a terminal state or not
        :param action: the action to execute
        :return: a list: state, value, reward, done
        """
        if self.state == 3 or self.state == 0 or self.state == 6:
            # if the state is terminal, dont do anything, just return done=True
            return self.state, self.curr_val, self.reward_matrix[self.state], True
        else:
            # Get the value to be added/subtracted corresponding to this action
            # from the action_to_value_mapping
            value = self.action_to_value_mapping[action]
            self.curr_val += value
            self.evaluate_state()
            done = False
            if self.state == 3 or self.state == 0 or self.state == 6:
                done = True
            return self.state, self.curr_val, self.reward_matrix[self.state], done

