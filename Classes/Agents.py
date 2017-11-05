from collections import deque

import numpy as np
import random
import itertools

import torch
from torch import nn
from torch.autograd import Variable


class Agent(object):

    def __init__(self, world, initial_action=None):
        if initial_action is None:
            self._last_action = random.randint(0, world.number_of_actions - 1)
        else:
            self._last_action = initial_action
        self._world = world

    @property
    def world(self):
        return self._world

    @property
    def last_action(self):
        return self._last_action

    def act(self, new_state, last_reward):
        pass

    def sample_action(self, state):
        pass


class QLearner(Agent):

    def __init__(self,
                 world,
                 initial_action=None,
                 alpha=0.2,
                 gamma=0.9,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.99):

        Agent.__init__(self, world=world, initial_action=initial_action)

        self._alpha = alpha
        self._gamma = gamma
        self._random_action_rate = random_action_rate
        self._random_action_decay_rate = random_action_decay_rate
        self._q_table = self.initiate_q_table()

    def initiate_q_table(self):

        q_table = dict()

        iterable_argument = \
            [range(self.world.state_space_bins_count + 1) for _ in range(self.world.number_of_features)]

        q_table_keys = list(itertools.product(*iterable_argument))

        q_table_initial_values = np.random.uniform(low=-1, high=1, size=(len(q_table_keys),
                                                                         self.world.number_of_actions))

        q_table.update(zip(q_table_keys, q_table_initial_values))

        return q_table

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
    def q_table(self):
        return self._q_table

    def q_value(self, reward, new_state, new_action):

        return (1 - self.alpha) * self.q_table[self.world.get_digitized_state_of_last_observation()][self.last_action] + self.alpha * (
            reward + self.gamma * self.q_table[new_state][new_action])

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
            chosen_action = np.argmax(self.q_table[new_state])

        self._random_action_rate *= self.random_action_decay_rate

        self._q_table[self.world.get_digitized_state_of_last_observation()][self.last_action] = self.q_value(
            reward=last_reward,
            new_state=new_state,
            new_action=chosen_action)

        self._last_action = chosen_action

        return chosen_action


class CartPoleAgent(QLearner):

    def __init__(self,
                 world,
                 initial_action=None,
                 alpha=0.2,
                 gamma=0.2,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.9):

        QLearner.__init__(self,
                          world=world,
                          initial_action=initial_action,
                          alpha=alpha,
                          gamma=gamma,
                          random_action_rate=random_action_rate,
                          random_action_decay_rate=random_action_decay_rate)


class DQNAgent(Agent):

    def __init__(self, world,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.99):

        Agent.__init__(self, world)

        self._random_action_rate = random_action_rate
        self._random_action_decay_rate = random_action_decay_rate
        self._module = None
        self._optimizer = None

        self.init_module()

    def init_module(self):

        self._module = nn.Module()

        self._module.__init__()

        self._module.f1 = nn.Linear(self.world.number_of_features, 200)
        self._module.relu = nn.ReLU()
        self._module.f2 = nn.Linear(200, self.world.number_of_actions)

        self._module.experience_replay = deque()
        self._module.action_num = self.world.number_of_actions

        self._module.batch_size = 16
        self._module.memory_size = 10000
        self._module.gamma = 0.9
        self._module.mse = nn.MSELoss()

        self._optimizer = torch.optim.Adam(self._module.parameters(), lr=1e-3)

    @property
    def optimizer(self):
        return self._optimizer

    def forward(self, x):

        out = self._module.f1(x)
        out = self._module.relu(out)
        out = self._module.f2(out)

        return out

    # def set_initial_state(self, state):
    #     """
    #     @summary: Sets the initial state and returns an action
    #     @param state: The initial state
    #     @returns: The selected action
    #     """
    #
    #     self._last_action = np.argmax(
    #         self.forward(Variable(torch.from_numpy(np.expand_dims(state, axis=0)).float())).data.numpy())

    def compute(self, new_state, reward):

        self._module.experience_replay.append(
            (self.world.last_observation, self.last_action, reward, new_state, self.world.is_done))

        if len(self._module.experience_replay) > self._module.memory_size:

            self._module.experience_replay.popleft()

        if len(self._module.experience_replay) > self._module.batch_size:

            self.train()

    def train(self):

        minibatch = random.sample(self._module.experience_replay, self._module.batch_size)

        state = [data[0] for data in minibatch]
        action = [data[1] for data in minibatch]
        reward = [data[2] for data in minibatch]
        new_state = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]

        y_label = []

        q_prime = self.forward(
            Variable(torch.from_numpy(np.array(new_state)).float())).data.numpy()

        # get the y_label e.t. the r+gamma*Q(s',a',w-)
        for i in xrange(self._module.batch_size):
            if done[i]:
                y_label.append(reward[i])
            else:
                y_label.append(reward[i] + np.max(q_prime[i]))

        # the input for the minibatch
        # Q(s,a,w)
        state_input = torch.from_numpy(np.array(state)).float()
        action_input = torch.from_numpy(np.array(action))
        out = self.forward(Variable(state_input))
        y_out = out.gather(1, Variable(action_input.unsqueeze(1)))

        self._optimizer.zero_grad()
        loss = self._module.mse(y_out, Variable(torch.from_numpy(np.array(y_label)).float()))
        loss.backward()
        self.optimizer.step()

    def act(self, new_state, last_reward):

        self.compute(new_state=new_state,
                     reward=last_reward)

        choose_random_action = (1 - self._random_action_rate) <= np.random.uniform(0, 1)

        self._random_action_rate *= self._random_action_decay_rate

        if choose_random_action:
            action = np.random.randint(self.world.number_of_actions)
        else:
            action = np.argmax(
                self.forward(Variable(torch.from_numpy(np.expand_dims(new_state, axis=0)).float())).data.numpy())

        self._last_action = action

        return action