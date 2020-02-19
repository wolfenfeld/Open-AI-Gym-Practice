import numpy as np
import random

from collections import deque

import torch
from torch import nn
from torch.autograd import Variable

from Modules.DecisionModels.BaseDecisionModel import BaseDecisionModel


class DQNModel(BaseDecisionModel):
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
        BaseDecisionModel.__init__(self)

        # The neural network module.
        self.neural_network = nn.Module()

        # Initializing the neural network module.
        self.neural_network.__init__()

        # The neural networks layers and activation function.
        self.neural_network.f1 = nn.Linear(number_of_features, 40)
        self.neural_network.f2 = nn.Linear(40, 100)
        self.neural_network.f3 = nn.Linear(100, number_of_actions)
        self.neural_network.relu = nn.ReLU()

        # The metric used to optimize.
        self.neural_network.mse = nn.MSELoss()
        # The optimizer.
        self.optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=1e-3)

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
        out = self.neural_network.f1(state)
        out = self.neural_network.relu(out)
        out = self.neural_network.f2(out)
        out = self.neural_network.relu(out)
        out = self.neural_network.f3(out)

        return out

    def compute(self, previous_state, previous_action, reward, state, done):
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
        Training the neural network with the sample.
        """

        # Sampling from the experience replay.
        sample = np.stack(random.sample(self.experience_replay, self.batch_size)).T

        state = np.stack(sample[0])
        action = sample[1].astype(int)
        reward = sample[2].astype(int)
        new_state = np.stack(sample[3])
        done = sample[4]

        # Estimating the Q-values with the forward passing over network with the new state.

        # new_state_tensor = torch.from_numpy(new_state).float()
        #
        # q = self.forward(Variable(new_state_tensor)).data.numpy()
        q = self.forward(Variable(torch.from_numpy(new_state).float())).data.numpy()
        # Computing y with the rewards and q.
        y = reward + (1-done) * self.gamma * np.max(q, axis=1)

        dtype = torch.FloatTensor

        y_tensor = torch.from_numpy(y.astype(float)).type(dtype)

        # Converting y to a torch variable,
        # target = Variable(y_tensor)
        target = Variable(y_tensor)

        # Computing the approximation of the target from the sample.
        s = torch.from_numpy(state).float()
        a = torch.from_numpy(action)

        approximation = self.forward(Variable(s)).gather(1, Variable(a.unsqueeze(1)))

        # Backward pass.
        self.optimizer.zero_grad()

        # Defining the loss - MSE(approximation, target).
        loss = self.neural_network.mse(approximation, target)
        loss.backward()
        self.optimizer.step()

    def update_model(self, previous_state, previous_action, reward, state, _, done):
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
        self.compute(previous_state, previous_action, reward, state, done)

    def get_action(self, state):
        """
        Get the action according to the DQN algorithm.
        :param state: the relevant state.
        :return: the action according to the DQN algorithm.
        """
        return np.argmax(self.forward(torch.from_numpy(state).float()).data.numpy())

    def get_random_action(self):
        """
        Get a random action.
        :return: a random action
        """

        return random.randint(0, self.number_of_actions - 1)
