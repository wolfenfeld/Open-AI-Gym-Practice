import numpy as np

from Modules.Agents.BaseAgents import BaseAgent
from Modules.DecisionModels.DQNDecisionModel import DQNModel
from Modules.DecisionModels.HillClimbDecisionModel import HillClimbModel
from Modules.DecisionModels.QTableDecisionModel import QTableModel


class HillClimbAgent(BaseAgent):

    def __init__(self,
                 world,
                 initial_action=None,
                 noise_scale=1e-1):
        """
        :param world: the world
        :param initial_action: the initial action
         """
        self.noise_scale = noise_scale

        BaseAgent.__init__(self, world=world, initial_action=initial_action)

        # Setting the decision module as a q-table.
        self.decision_model = HillClimbModel(number_of_features=world.number_of_features,
                                             number_of_actions=world.number_of_actions,
                                             alpha=noise_scale)

    def is_random_action(self):

        return False


class QLearnerAgent(BaseAgent):
    """
    Q-Learner agent - uses the q-learning algorithm to play.
    """
    def __init__(self,
                 world,
                 initial_action=None,
                 alpha=0.5,
                 gamma=0.95,
                 random_action_rate=1,
                 random_action_decay_rate=0.95):
        """
        :param world: the world
        :param initial_action: the initial action
        :param alpha: the alpha parameter for the q-function update rule
        :param gamma: the gamma parameter (discount factor) for the q-function update rule.
        :param random_action_rate: defines the amount of random actions taken
        :param random_action_decay_rate: the decay rate of the random action - reducing the amount of random actions.
        """
        self.random_action_rate = random_action_rate
        self.random_action_decay_rate = random_action_decay_rate

        BaseAgent.__init__(self, world=world, initial_action=initial_action)

        # Setting the decision module as a q-table.
        self.decision_model = QTableModel(
            number_of_discrete_values_per_feature=world.number_of_discrete_values_per_feature,
            number_of_features=world.number_of_features,
            number_of_actions=world.number_of_actions,
            alpha=alpha,
            gamma=gamma)

    def is_random_action(self):

        # Choosing a random action with probability of random_action_rate.
        result = (1 - self.random_action_rate) <= np.random.uniform(0, 1)

        # Reducing the probability of choosing a random action.
        self.random_action_rate *= self.random_action_decay_rate

        return result


class DQNAgent(BaseAgent):
    """
    Agent object using the DQN algorithm.
    """
    def __init__(self, world):
        """
        Initializing the agent.
        :param world: the world that the agent will play in.
        """
        BaseAgent.__init__(self, world)

        # The value used for the epsilon-greedy method.
        self.epsilon = 1.0

        # The DQN decision module.
        self.decision_model = DQNModel(number_of_features=world.number_of_features,
                                       number_of_actions=world.number_of_actions,
                                       gamma=0.99,
                                       batch_size=128,
                                       memory_size=5000)

    def is_random_action(self):
        # The random action probability
        self.epsilon /= self.number_of_episodes_played

        return (1 - self.epsilon) <= np.random.uniform(0, 1)
