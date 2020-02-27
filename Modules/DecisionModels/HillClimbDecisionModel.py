import random
import itertools
import numpy as np

from Modules.DecisionModels.BaseDecisionModel import BaseDecisionModel


class HillClimbModel(BaseDecisionModel):
    """
    Q-table module - using a Q-learning policy to return an action.
    """
    def __init__(self, number_of_discrete_values_per_feature, number_of_features, number_of_actions, alpha=1e-2):
        """
        Initializing the Q-learning module that holds a state-action pair table with all the estimated Q-values.
        There are number_of_discrete_values_per_feature*number_of_features states and number_of_actions actions..
        :param number_of_discrete_values_per_feature: used to index table.
        :param number_of_features: used to index table
        :param number_of_actions: used to index table
        """

        self.number_of_discrete_values_per_feature = number_of_discrete_values_per_feature
        self.number_of_features = number_of_features
        self.number_of_actions = number_of_actions

        self.weights = 1e-4 * np.random.rand(self.number_of_features, self.number_of_actions)
        self.best_reward = -np.Inf
        self.best_weights = np.copy(self.weights)
        self.alpha = alpha

        BaseDecisionModel.__init__(self)

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
        if reward >= self.best_reward:
            self.best_reward = reward
            self.best_weights = np.copy(self.weights)
            self.alpha = max(self.alpha / 1.5, 1e-2)
        else:
            self.alpha = min(self.alpha * 2, 2)

        self.weights = self.best_weights + self.alpha * np.random.rand(self.number_of_features, self.number_of_actions)

    def get_action(self, state):
        """
        Getting the action with the highest Q-value for a state
        :param state: the state.
        :return: the action with the highest Q-value for a state.
        """

        return np.argmax(np.dot(state, self.weights))

    def get_random_action(self):
        """
        Getting a random action.
        :return:a random action.
        """
        return random.randint(0, self.number_of_actions - 1)

