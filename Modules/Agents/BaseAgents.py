import random

from Modules.DecisionModels.BaseDecisionModel import Transition


class BaseAgent(object):
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
        self.decision_model = None

    def is_random_action(self):

        raise NotImplementedError

    def get_action(self, state):
        """
        Sampling an action.
        :param state: agent state
        :return:
        """

        if self.is_random_action():
            # Sampling a random action.
            action = self.decision_model.get_random_action()
        else:
            # Sampling an action according to the DQN algorithm.
            action = self.decision_model.get_action(state)

        return action

    def reinforce(self, transition: Transition):
        """
        Updating the agent
        :param transition: state, action, reward, new_state, done
        """

        # Updating the decision module.
        self.decision_model.update_model(transition)

        # # Updating the last action.
        # self.last_action = action
