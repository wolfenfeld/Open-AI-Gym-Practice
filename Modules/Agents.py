from Modules.DecisionModules import QTableModule, DQNModule

import numpy as np
import random


class Agent(object):
    """
    The agent object that will play the game.
    """
    def __init__(self, world, initial_action=None):
        """
        Initializing the agent object
        :param world: the world where the agent will play in.
        :param initial_action: the initial action - by default set to a random set.
        """

        # Setting the initial action.
        if initial_action is None:
            self.last_action = random.randint(0, world.number_of_actions - 1)
        else:
            self.last_action = initial_action

        # Setting the world.
        self.world = world

        # The number of episodes played.
        self.number_of_episodes_played = 1

        # The decision module of the agent.
        self.decision_module = None

    def is_random_action(self):

        raise NotImplementedError

    def sample_action(self, state):
        """
        Sampling an action.
        :param state:
        :return:
        """

        if self.is_random_action():
            # Sampling a random action.
            action = self.decision_module.get_random_action()
        else:
            # Sampling an action according to the DQN algorithm.
            action = self.decision_module.get_action(state)

        return action

    def update_agent(self, state, action, reward, episode, done):
        """
        Updating the agent
        :param state: the state
        :param action: the action
        :param reward: the reward
        :param episode: the episode played
        :param done: the status of the game
        """
        # The number of episodes played.
        self.number_of_episodes_played = episode

        # Updating the decision module.
        self.decision_module.update_module(self.world.last_observation, self.last_action, reward, state, action, done)

        # Updating the last action.
        self.last_action = action


class QLearnerAgent(Agent):
    """
    Q-Learner agent - uses the q-learning algorithm to play.
    """
    def __init__(self,
                 world,
                 initial_action=None,
                 alpha=0.2,
                 gamma=1,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.9):
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

        Agent.__init__(self, world=world, initial_action=initial_action)

        # Setting the decision module as a q-table.
        self.decision_module = QTableModule(
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


class DQNAgent(Agent):
    """
    Agent object using the DQN algorithm.
    """
    def __init__(self, world):
        """
        Initializing the agent.
        :param world: the world that the agent will play in.
        """
        Agent.__init__(self, world)

        # The value used for the epsilon-greedy method.
        self.epsilon = 1.0

        # The DQN decision module.
        self.decision_module = DQNModule(number_of_features=world.number_of_features,
                                         number_of_actions=world.number_of_actions,
                                         gamma=0.9,
                                         batch_size=16,
                                         memory_size=10000)

    def is_random_action(self):
        # The random action probability
        self.epsilon /= self.number_of_episodes_played

        return (1 - self.epsilon) <= np.random.uniform(0, 1)
