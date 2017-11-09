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
            self.last_action = random.randint(0, world.number_of_actions - 1)
        else:
            self.last_action = initial_action

        self.world = world

        self.number_of_episodes_played = 1

    def sample_action(self, current_state):
        pass

    def update_agent(self, state, action, reward, episode):
        pass


class QLearnerAgent(Agent):

    def __init__(self,
                 world,
                 initial_action=None,
                 alpha=0.2,
                 gamma=1,
                 random_action_rate=0.5,
                 random_action_decay_rate=0.9):

        Agent.__init__(self, world=world, initial_action=initial_action)

        self.alpha = alpha
        self.gamma = gamma
        self.random_action_rate = random_action_rate
        self.random_action_decay_rate = random_action_decay_rate
        self.q_table = self.initiate_q_table()

    def initiate_q_table(self):

        q_table = dict()

        iterable_argument = \
            [range(self.world.number_of_discrete_values_per_feature + 1) for _ in range(self.world.number_of_features)]

        q_table_keys = list(itertools.product(*iterable_argument))

        q_table_initial_values = np.random.uniform(low=-1, high=1, size=(len(q_table_keys),
                                                                         self.world.number_of_actions))

        q_table.update(zip(q_table_keys, q_table_initial_values))

        return q_table

    def sample_action(self, current_state):
        """
        @summary: Moves to the given state with given reward and returns action
        @param current_state: The current state
        @returns: The selected action
        """

        choose_random_action = (1 - self.random_action_rate) <= np.random.uniform(0, 1)

        if choose_random_action:
            chosen_action = random.randint(0, self.world.number_of_actions - 1)
        else:
            chosen_action = np.argmax(self.q_table[current_state])

        self.random_action_rate *= self.random_action_decay_rate

        return chosen_action

    def q_value(self, state, action, reward):

        return (1 - self.alpha) * self.q_table[self.world.last_observation][self.last_action] + \
               self.alpha * (reward + self.gamma * self.q_table[state][action])

    def update_agent(self, state, action, reward, episode):

        self.number_of_episodes_played = episode

        self.q_table[self.world.last_observation][self.last_action] = self.q_value(
            state=state,
            action=action,
            reward=reward)

        self.last_action = action


class DQNAgent(Agent):

    def __init__(self, world):

        Agent.__init__(self, world)

        self.epsilon = 1.0

        self.module = None
        self.optimizer = None

        self.init_module()

    def init_module(self):

        self.module = nn.Module()

        self.module.__init__()

        self.module.f1 = nn.Linear(self.world.number_of_features, 200)
        self.module.relu = nn.ReLU()
        self.module.f2 = nn.Linear(200, self.world.number_of_actions)

        self.module.experience_replay = deque()
        self.module.action_num = self.world.number_of_actions

        self.module.batch_size = 16
        self.module.memory_size = 10000
        self.module.gamma = 0.9
        self.module.mse = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)

    def forward(self, x):

        out = self.module.f1(x)
        out = self.module.relu(out)
        out = self.module.f2(out)

        return out

    def compute(self, new_state, reward):

        self.module.experience_replay.append(
            (self.world.last_observation, self.last_action, reward, new_state, self.world.is_done))

        if len(self.module.experience_replay) > self.module.memory_size:

            self.module.experience_replay.popleft()

        if len(self.module.experience_replay) > self.module.batch_size:

            self.train()

    def train(self):

        minibatch = random.sample(self.module.experience_replay, self.module.batch_size)

        state = [data[0] for data in minibatch]
        action = [data[1] for data in minibatch]
        reward = [data[2] for data in minibatch]
        new_state = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]

        y_label = []

        q_prime = self.forward(
            Variable(torch.from_numpy(np.array(new_state)).float())).data.numpy()

        # get the y_label e.t. the r+gamma*Q(s',a',w-)
        for i in range(self.module.batch_size):
            if done[i]:
                y_label.append(reward[i])
            else:
                y_label.append(reward[i] + self.module.gamma*np.max(q_prime[i]))

        # the input for the minibatch
        # Q(s,a,w)
        state_input = torch.from_numpy(np.array(state)).float()
        action_input = torch.from_numpy(np.array(action))
        out = self.forward(Variable(state_input))
        y_out = out.gather(1, Variable(action_input.unsqueeze(1)))

        self.optimizer.zero_grad()
        loss = self.module.mse(y_out, Variable(torch.from_numpy(np.array(y_label)).float()))
        loss.backward()
        self.optimizer.step()

    def sample_action(self, state):

        self.epsilon /= self.number_of_episodes_played

        choose_random_action = (1-self.epsilon) <= np.random.uniform(0, 1)

        if choose_random_action:
            action = np.random.randint(self.world.number_of_actions)
        else:
            action = np.argmax(
                self.forward(Variable(torch.from_numpy(np.expand_dims(state, axis=0)).float())).data.numpy())

        return action

    def update_agent(self, state, action, reward, episode):

        self.number_of_episodes_played = episode

        self.compute(state, reward)

        self.last_action = action
