import gym
import numpy as np


class World(object):

    def __init__(self, env_name='', max_episode_steps=200, **kwargs):

        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.env._max_episode_steps = max_episode_steps
        self.last_observation = None
        self.last_reward = 0
        self.is_done = None
        self.last_info = None
        self.number_of_actions = self.env.action_space

        for k, v in kwargs.items():
            setattr(self, '_'+k, v)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def interact_with_world(self, action):
        pass


class DiscreteWorld(World):

    def __init__(self,
                 env_name,
                 max_episode_steps=200,
                 number_of_discrete_values_per_feature=10,
                 **kwargs):

        World.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        self.number_of_discrete_values_per_feature = number_of_discrete_values_per_feature


class DiscreteCartPoleWorld(DiscreteWorld):

    def __init__(self,
                 env_name='CartPole-v0',
                 max_episode_steps=200,
                 **kwargs):

        DiscreteWorld.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        self.number_of_actions = self.env.action_space.n

        self.features = ['cart_position', 'pole_angle', 'cart_velocity', 'angle_rate']
        self.number_of_features = len(self.features)

        self.cart_position_bins = np.linspace(start=-3, stop=3, num=self.number_of_discrete_values_per_feature)
        self.pole_angle_bins = np.linspace(start=-5, stop=5, num=self.number_of_discrete_values_per_feature)
        self.cart_velocity_bins = np.linspace(start=-1, stop=1, num=self.number_of_discrete_values_per_feature)
        self.angle_rate_bins = np.linspace(start=-5, stop=5, num=self.number_of_discrete_values_per_feature)

    def reset(self):

        self.last_observation = self.digitize_step(self.env.reset())

        return self.last_observation

    def digitize_step(self, state):
        digitized_state = list()

        for index, feature in enumerate(self.features):
            digitized_state.append(np.digitize(x=[state[index]],
                                               bins=getattr(self, feature + '_bins'))[0])

        return tuple(digitized_state)

    def interact_with_world(self, action):

        state, reward, done, _ = self.step(action)

        return self.digitize_step(state), reward, done


class CartPoleWorld(World):

    def __init__(self,
                 env_name='CartPole-v0',
                 max_episode_steps=200,
                 **kwargs):

        World.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        self.number_of_actions = self.env.action_space.n

        self.features = ['cart_position', 'pole_angle', 'cart_velocity', 'angle_rate']
        self.number_of_features = len(self.features)

    def reset(self):

        self.last_observation = self.env.reset()

        return self.last_observation

    def interact_with_world(self, action):

        state, reward, done, _ = self.step(action)

        return state, reward, done
