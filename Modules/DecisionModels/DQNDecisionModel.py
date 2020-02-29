import numpy as np
import random

import torch
from torch import nn

from Modules.DecisionModels.BaseDecisionModel import BaseDecisionModel, Transition


class DQNModel(BaseDecisionModel):
    """
        Q-table module - using the DQN algorithm to return an action.
        """
    def __init__(self,
                 number_of_features,
                 number_of_actions,
                 gamma,
                 batch_size,
                 memory_size):
        """
        Initializing the DQN module
        :param number_of_features: number of features.
        :param number_of_actions: number of actions
        :param gamma: discount factor
        :param batch_size: the batch size for training the network.
        :param memory_size: the memory size (where all the feedback and actions are stored)
        """
        BaseDecisionModel.__init__(self)

        self.q_network = QNetwork(number_of_features, number_of_actions)

        # The optimizer.
        self.optimizer = torch.optim.Adam(self.q_network.parameters, lr=1e-3)

        # The depth of the experience_replay.
        self.memory_size = memory_size

        # The memory where all the experience is stored (and used to train the network).
        self.replay_memory = ReplayMemory(memory_size)

        # The number of possible actions.
        self.number_of_actions = number_of_actions

        # The batch size for the training
        self.batch_size = batch_size

        # The discount factor.
        self.gamma = gamma

    def update_model(self, transition: Transition):
        """
        Updating the experience_replay memory and training the network.
        :param transition: transaction data - state, action, reward, next_state, done
        :
        """

        # Updating experience_replay.
        self.replay_memory.push(transition)

        # Training if we have enough examples.
        if len(self.replay_memory) > self.batch_size:
            self.train()

    def train(self):
        """
        Training the neural network with the sample.
        """

        # Sampling from the experience replay.
        state, action, reward, new_state, done = self.replay_memory.sample(self.batch_size)

        # Computing q values.
        q_values = self.q_network.compute_q_values(new_state, reward, done, self.gamma)

        # Computing the q value approximation.
        predicted_q_values = self.q_network.predict_q_values(state, action)

        # reset gradient.
        self.optimizer.zero_grad()

        # Compute the loss.
        loss = self.q_network.compute_loss(predicted_q_values, q_values)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        """
        Get the action according to the DQN algorithm.
        :param state: the relevant state.
        :return: the action according to the DQN algorithm.
        """
        return self.q_network.get_action_with_max_value(state)

    def get_random_action(self):
        """
        Get a random action.
        :return: a random action
        """

        return random.randint(0, self.number_of_actions - 1)


class QNetwork(object):

    def __init__(self, number_of_features, number_of_actions):
        # The neural network module.
        self.neural_network = nn.Module()

        # Initializing the neural network module.
        self.neural_network.__init__()

        # The neural networks layers and activation function.
        self.neural_network.f1 = nn.Linear(number_of_features, 32)
        self.neural_network.f2 = nn.Linear(32, 64)
        self.neural_network.f3 = nn.Linear(64, number_of_actions)
        self.neural_network.relu = nn.ReLU()

        # The metric used to optimize.
        self.neural_network.mse = nn.MSELoss()

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

    def compute_q_values(self, new_state, reward, done, gamma):

        values_from_new_state = self._get_values_for_state(new_state)

        reward = reward*(1-done) - 100*done*torch.ones_like(reward)

        return reward + gamma*(1-done) * values_from_new_state

    def _get_values_for_state(self, state):

        return torch.max(self.forward(state), dim=1)[0]

    def predict_q_values(self, state, action):

        return self.forward(state).gather(1, action.unsqueeze(1))

    def get_action_with_max_value(self, state):

        state_tensor = torch.from_numpy(state).float()

        return np.argmax(self.forward(state_tensor).data.numpy())

    @property
    def parameters(self):
        return self.neural_network.parameters()

    def compute_loss(self, approximation, target):

        return self.neural_network.mse(approximation, target)


class ReplayMemory(object):

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = list()
        self.position = 0

    def push(self, transition: Transition):

        if len(self.memory) < self.memory_size:
            self.memory.append(None)

        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.memory_size

    def sample(self, batch_size):

        random_sample = Transition(*zip(*random.sample(self.memory, batch_size)))

        state = np.stack(random_sample.state)
        action = np.array(random_sample.action).astype(int)
        reward = np.array(random_sample.reward).astype(int)
        next_state = np.stack(random_sample.next_state)
        done = np.array(random_sample.done).astype(int)

        state_tensor = torch.from_numpy(state).float()
        action_tensor = torch.from_numpy(action)
        next_state_tensor = torch.from_numpy(next_state).float()
        reward_tensor = torch.from_numpy(reward).float()
        done_tensor = torch.from_numpy(done).float()

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor

    def __len__(self):
        return len(self.memory)
