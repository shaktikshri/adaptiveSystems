import numpy as np


def describe_policy_matrix(matrix, env):
    for state, action in enumerate(matrix):
        if action == -1:
            print('Terminal State ', state, ' : No Action')
        else:
            value = env.action_to_value_mapping[action]
            if value > 0:
                string = 'Subtract '+str(abs(value))+' from current value'
            else:
                string = 'Add '+str(abs(value))+' to current value'
            print('State ', state, ' : Action : ', string)


def get_return(state_list, gamma):
    """
    :param state_list: a list of tuples (observation, action, reward)
    :param gamma: the discount factor
    :return: the return value for that state_list
    """
    return_value = 0
    counter = 0
    for visit in state_list:
        reward = visit[2]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value


def update_policy(episode_list, policy_matrix, state_action_matrix):
    """
    Updates the policy in a greedy way, selecting actions which have the highest
    utility for each state visited in the episode_list
    :param episode_list: the tuples of states visited as (observation, action, reward)
    :param policy_matrix: the policy matrix
    :param state_action_matrix: the Q matrix
    :return:
    """
    for visit in episode_list:
        state = visit[0][0]
        if policy_matrix[state] != -1:
            # if its not the terminal state
            policy_matrix[state] = np.argmax(state_action_matrix[:, state])
    return policy_matrix


def has_converged(old_matrix, new_matrix):
    if abs(np.sum(new_matrix-old_matrix)) < 0.1:
        return True
    return False
