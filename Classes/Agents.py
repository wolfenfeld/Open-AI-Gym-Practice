import numpy as np
import random
import itertools


class Agent(object):

    def __init__(self, world=None, initial_action=None, initial_state=None):
        self._last_action = initial_action
        self._current_state = initial_state
        self._world = world

    @property
    def world(self):
        return self._world

    @property
    def last_action(self):
        return self._last_action

    @property
    def current_state(self):
        return self._current_state

    def update_current_state(self, new_state):
        self.current_state(new_state)

    # @staticmethod
    # def encode_state(state):
    #     pass
    #
    # @staticmethod
    # def decode_encoded_state(encoded_state):
    #     pass
    #
    # @staticmethod
    # def encode_action(action):
    #     pass
    #
    # @staticmethod
    # def decode_encoded_action(encoded_action):
    #     pass


class QLearner(Agent):

    def __init__(self,
                 world,
                 initial_action=None,
                 initial_state=None,
                 alpha=0.2,
                 gamma=0.9,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.99):

        Agent.__init__(self, initial_action=initial_action, initial_state=initial_state)

        self._world = world
        self._alpha = alpha
        self._gamma = gamma
        self._random_action_rate = random_action_rate
        self._random_action_decay_rate = random_action_decay_rate
        self._qtable = self.initiate_qtable()

    def initiate_qtable(self):

        qtable = dict()

        iterable_argument = \
            [range(self.world.state_space_bins_count+1) for _ in range(self.world.number_of_features)] + \
            [range(self.world.number_of_actions)]

        qtable_keys = list(itertools.product(*iterable_argument))

        qtable_initial_values = np.random.uniform(low=-1, high=1, size=(len(qtable_keys))).tolist()

        qtable.update(zip(qtable_keys, qtable_initial_values))

        return qtable

    @property
    def alpha(self):
        return self._alpha

    @property
    def gamma(self):
        return self._gamma

    @property
    def random_action_rate(self):
        return self._random_action_rate

    @property
    def random_action_decay_rate(self):
        return self._random_action_decay_rate

    @property
    def qtable(self):
        return self._qtable

    def set_initial_state(self, state):
        """
        @summary: Sets the initial state and returns an action
        @param state: The initial state
        @returns: The selected action
        """
        self._current_state = state

        self._last_action = self.get_best_action()

        return self._last_action

    def get_best_action(self):

        best_action = 0
        best_qvalue = -1

        for action_number in range(self.world.number_of_actions):

            state_action_pair = self.current_state + (action_number,)
            if self.qtable[state_action_pair] > best_qvalue:
                best_qvalue = self.qtable[state_action_pair]
                best_action = action_number

        return best_action

    def q_value(self, reward, state, action):

        current_state_action_pair = self.current_state + (self.last_action,)
        new_state_action_pair = state + (action,)

        return (1 - self.alpha) * self.qtable[current_state_action_pair] + self.alpha * (
            reward + self.gamma * self.qtable[new_state_action_pair])

    def act(self, new_state, last_reward):
        """
        @summary: Moves to the given state with given reward and returns action
        @param new_state: The new state
        @param last_reward: The reward
        @returns: The selected action
        """

        choose_random_action = (1 - self.random_action_rate) <= np.random.uniform(0, 1)

        if choose_random_action:
            chosen_action = random.randint(0, self.world.number_of_actions - 1)
        else:
            chosen_action = self.get_best_action()

        self._random_action_rate *= self.random_action_decay_rate

        state_action_pair = self.current_state+(self.last_action,)

        self._qtable[state_action_pair] = self.q_value(
            reward=last_reward,
            state=new_state,
            action=chosen_action)

        self._last_action = chosen_action
        self._current_state = new_state

        return chosen_action


class CartPoleAgent(QLearner):

    def __init__(self,
                 world,
                 initial_action=None,
                 initial_state=None,
                 alpha=0.2,
                 gamma=0.9,
                 random_action_rate=0.7,
                 random_action_decay_rate=0.99):

        QLearner.__init__(self,
                          world=world,
                          initial_action=initial_action,
                          initial_state=initial_state,
                          alpha=alpha,
                          gamma=gamma,
                          random_action_rate=random_action_rate,
                          random_action_decay_rate=random_action_decay_rate)

    # @staticmethod
    # def encode_action(action):
    #     return tuple(action)
    #
    # @staticmethod
    # def encode_state(state):
    #     return tuple(i for i in state)
