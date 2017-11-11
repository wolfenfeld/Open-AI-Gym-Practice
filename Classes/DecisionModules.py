import itertools
import numpy as np
import random

from collections import deque

import torch
from torch import nn
from torch.autograd import Variable


class DecisionModule(object):
    """
    A decision model of and agents provides an action according to its policy.
    """
    def __init__(self):
        pass

    def get_action(self, state):
        """
        Getting an action according to the modules policy.
        :param state: a state.
        :return: the chosen action according to the modules policy.
        """
        pass

    def get_random_action(self):
        """
        Returning a random action.
        :return: a random action.
        """
        pass


class QTableModule(DecisionModule):
    """
    Q-table module - using a Q-learning policy to return an action.
    """
    def __init__(self,
                 number_of_discrete_values_per_feature,
                 number_of_features,
                 number_of_actions,
                 alpha,
                 gamma):
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
        self.q_table = self.initiate_q_table()

        DecisionModule.__init__(self)

    def initiate_q_table(self):
        """
        Initializing the Q-table with random values.
        :return: the Q-table - a number_of_discrete_values_per_feature * number_of_discrete_values_per_feature by
        number_of_actions table.
        """
        q_table = dict()

        iterable_argument = \
            [range(self.number_of_discrete_values_per_feature + 1) for _ in range(self.number_of_features)]

        # The index of the table
        q_table_keys = list(itertools.product(*iterable_argument))

        # The initial random values
        q_table_initial_values = np.random.uniform(low=-1, high=1, size=(len(q_table_keys),
                                                                         self.number_of_actions))
        # Setting the values.
        q_table.update(zip(q_table_keys, q_table_initial_values))

        return q_table

    def q_value(self, state, action, reward, previous_state, previous_action):
        """
        Calculating the Q-value.
        :param state: the current state
        :param action: the current action
        :param reward: the current reward
        :param previous_state: the previous state
        :param previous_action: the previous action.
        :return: The Q-value
        """

        return (1 - self.alpha) * self.q_table[previous_state][previous_action] + \
            self.alpha * (reward + self.gamma * self.q_table[state][action])

    def update_module(self, state, action, reward, previous_state, previous_action, done):
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
        self.q_table[previous_state][previous_action] = self.q_value(
            state=state,
            action=action,
            reward=reward,
            previous_state=previous_state,
            previous_action=previous_action)

    def get_action(self, state):
        """
        Getting the action with the highest Q-value for a state
        :param state: the state.
        :return: the action with the highest Q-value for a state.
        """
        return np.argmax(self.q_table[state])

    def get_random_action(self):
        """
        Getting a random action.
        :return:a random action.
        """
        return random.randint(0, self.number_of_actions - 1)


class DQNModule(DecisionModule):
    """
        Q-table module - using the DQN algorithm to return an action.
        """
    def __init__(self,
                 number_of_features,
                 number_of_actions,
                 gamma=0.9,
                 batch_size=16,
                 memory_size=10000):
        """
        Initializing the DQN module
        :param number_of_features: number of features.
        :param number_of_actions: number of actions
        :param gamma: discount factor
        :param batch_size: the batch size for training the network.
        :param memory_size: the memory size (where all the feedback and actions are stored)
        """
        DecisionModule.__init__(self)

        # The neural network module.
        self.module = nn.Module()

        # Initializing the neural network module.
        self.module.__init__()

        # The neural networks layers and activation function.
        self.module.f1 = nn.Linear(number_of_features, 200)
        self.module.relu = nn.ReLU()
        self.module.f2 = nn.Linear(200, number_of_actions)

        # The metric used to optimize.
        self.module.mse = nn.MSELoss()
        # The optimizer.
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)

        # The memory where all the experience is stored (and used to train the network).
        self.experience_replay = deque()
        # The depth of the experience_replay.
        self.memory_size = memory_size

        # The number of possible actions.
        self.number_of_actions = number_of_actions

        # The batch size for the training
        self.batch_size = batch_size

        # The discount factor.
        self.gamma = gamma

    def forward(self, state):
        """
        Forward pass of the neural network.
        :param state: the state
        :return: the estimate of Q for the state.
        """
        out = self.module.f1(state)
        out = self.module.relu(out)
        out = self.module.f2(out)

        return out

    def compute(self, state, reward, previous_state, previous_action, done):
        """
        Updating the experience_replay memory and training the network.
        :param state: current state
        :param reward: current reward
        :param previous_state: previous state
        :param previous_action: previous action
        :param done: done status
        """
        # Updating experience_replay.
        self.experience_replay.append(
            (previous_state, previous_action, reward, state, done))

        # Popping experience out if out of memory.
        if len(self.experience_replay) > self.memory_size:
            self.experience_replay.popleft()

        # Training if we have enough examples.
        if len(self.experience_replay) > self.batch_size:
            self.train()

    def train(self):
        """
        Training the neural network with the minibatch.
        """

        # Sampling from the experience replay.
        minibatch = random.sample(self.experience_replay, self.batch_size)

        state = [data[0] for data in minibatch]
        action = [data[1] for data in minibatch]
        reward = [data[2] for data in minibatch]
        new_state = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]

        # Estimating the Q-values with the forward passing over network with the new state.
        q_prime = self.forward(
            Variable(torch.from_numpy(np.array(new_state)).float())).data.numpy()

        # Computing y_label with the rewards and q_prime.
        y_label = []

        for i in range(self.batch_size):
            if done[i]:
                y_label.append(reward[i])
            else:
                y_label.append(reward[i] + self.gamma * np.max(q_prime[i]))

        # Computing y_out from the minibatch.
        state_input = torch.from_numpy(np.array(state)).float()
        action_input = torch.from_numpy(np.array(action))
        out = self.forward(Variable(state_input))
        y_out = out.gather(1, Variable(action_input.unsqueeze(1)))

        # Backward pass.
        self.optimizer.zero_grad()
        # Defining the loss - MSE(y_out, y_label).
        loss = self.module.mse(y_out, Variable(torch.from_numpy(np.array(y_label)).float()))
        loss.backward()
        self.optimizer.step()

    def update_module(self, state, _, reward, previous_state, previous_action, done):
        """
        Updating the module.
        :param state: the current state.
        :param _: placeholder from the current action.
        :param reward: the current reward.
        :param previous_state: the previous state.
        :param previous_action: the previous action.
        :param done: the done indicator.
        """
        # If done inflict a -200 fine.
        if done:
            reward = -200

        # Update the network.
        self.compute(state, reward, previous_state, previous_action, done)

    def get_action(self, state):
        """
        Get the action according to the DQN algorithm.
        :param state: the relevant state.
        :return: the action according to the DQN algorithm.
        """

        return np.argmax(self.forward(Variable(torch.from_numpy(np.expand_dims(state, axis=0)).float())).data.numpy())

    def get_random_action(self):
        """
        Get a random action.
        :return: a random action
        """

        return random.randint(0, self.number_of_actions - 1)
