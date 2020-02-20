import random
import itertools
import numpy as np

from Modules.DecisionModels.BaseDecisionModel import BaseDecisionModel


class QTableModel(BaseDecisionModel):
    """
    Q-table module - using a Q-learning policy to return an action.
    """
    def __init__(self, number_of_discrete_values_per_feature, number_of_features, number_of_actions, alpha, gamma):
        """
        Initializing the Q-learning module that holds a state-action pair table with all the estimated Q-values.
        There are number_of_discrete_values_per_feature*number_of_features states and number_of_actions actions..
        :param number_of_discrete_values_per_feature: used to index table.
        :param number_of_features: used to index table
        :param number_of_actions: used to index table
        :param alpha: the alpha parameter for calculating the Q-values.
        :param gamma: the discount factor for calculating the Q-values.
        """

        self.number_of_discrete_values_per_feature = number_of_discrete_values_per_feature
        self.number_of_features = number_of_features
        self.number_of_actions = number_of_actions
        self.alpha = alpha
        self.gamma = gamma

        # Initializing the q table.
        self.q_table = QTable(number_of_discrete_values_per_feature, number_of_features, number_of_actions)

        BaseDecisionModel.__init__(self)

    def _q_value(self, previous_state, previous_action, reward, state, action):
        """
        Calculating the Q-value.
        :param state: the current state
        :param action: the current action
        :param reward: the current reward
        :param previous_state: the previous state
        :param previous_action: the previous action.
        :return: The Q-value
        """

        return (1 - self.alpha) * self.q_table.get_value(previous_state, previous_action) + \
            self.alpha * (reward + self.gamma * self.q_table.get_value(state, action))

    def update_model(self, previous_state, previous_action, reward, state, action, done):
        """
        Updating the module.
        :param state: the current state
        :param action: the current action
        :param reward: the current reward
        :param previous_state: the previous state
        :param previous_action: the previous action.
        :param done: episode is done indicator.
        """
        # The if the episode is done a -200 fine is inflicted.
        if done:
            reward = -200

        # Updating the Q-table.
        self.q_table.set_value(previous_state, previous_action,
                               self._q_value(previous_state, previous_action, reward, state, action))

    def get_action(self, state):
        """
        Getting the action with the highest Q-value for a state
        :param state: the state.
        :return: the action with the highest Q-value for a state.
        """
        return self.q_table.get_action_with_max_value(state)

    def get_random_action(self):
        """
        Getting a random action.
        :return:a random action.
        """
        return random.randint(0, self.number_of_actions - 1)


class QTable(object):

    def __init__(self, number_of_discrete_values_per_feature, number_of_features, number_of_actions):
        self.table = dict()

        iterable_argument = \
            [range(number_of_discrete_values_per_feature + 1) for _ in range(number_of_features)]

        # The index of the table
        q_table_keys = list(itertools.product(*iterable_argument))

        # The initial random values
        q_table_initial_values = np.random.uniform(low=-1, high=1, size=(len(q_table_keys),
                                                                         number_of_actions))
        # Setting the values.
        self.table.update(zip(q_table_keys, q_table_initial_values))

    def get_value(self, state, action):

        return self.table[state][action]

    def set_value(self, state, action, value):

        self.table[state][action] = value

    def get_action_with_max_value(self, state):

        return np.argmax(self.table[state])
