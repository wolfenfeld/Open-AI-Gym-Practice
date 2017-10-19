import gym
import numpy as np


class World(object):

    def __init__(self, env_name='', max_episode_steps=200, **kwargs):

        self._env_name = env_name
        self._env = gym.make(self.env_name)
        self._env._max_episode_steps = max_episode_steps
        self._last_observation = None
        self._last_reward = None
        self._is_done = None
        self._last_info = None

        for k, v in kwargs.iteritems():
            setattr(self, '_'+k, v)

    @property
    def env_name(self):
        return self._env_name

    @property
    def env(self):
        return self._env

    @property
    def last_observation(self):
        return self._last_observation

    @property
    def last_reward(self):
        return self._last_reward

    @property
    def is_done(self):
        return self._is_done

    @property
    def last_info(self):
        return self._last_info

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self.env.step(action)

    def interact_with_world(self, action):
        self._last_observation, self._last_reward, self._is_done, self._last_info = self._env.step(action)

    def render(self):
        self._env.render()


class QLearningDiscreteWorld(World):

    def __init__(self,
                 env_name,
                 max_episode_steps=200,
                 state_space_bins_count=10,
                 **kwargs):

        World.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        self._state_space_bins_count = state_space_bins_count

    @property
    def state_space_bins_count(self):
        return self._state_space_bins_count


class CartPoleWorld(QLearningDiscreteWorld):

    def __init__(self,
                 env_name='CartPole-v0',
                 max_episode_steps=200,
                 **kwargs):

        QLearningDiscreteWorld.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        self._number_of_actions = self._env.action_space.n

        self._features = ['cart_position', 'pole_angle', 'cart_velocity', 'angle_rate']
        self._number_of_features = len(self._features)

        self._cart_position_bins = np.linspace(start=-3, stop=3, num=self.state_space_bins_count+1)
        self._pole_angle_bins = np.linspace(start=-5, stop=5, num=self.state_space_bins_count+1)
        self._cart_velocity_bins = np.linspace(start=-1, stop=1, num=self.state_space_bins_count+1)
        self._angle_rate_bins = np.linspace(start=-5, stop=5, num=self.state_space_bins_count+1)

    @property
    def number_of_actions(self):
        return self._number_of_actions

    @property
    def number_of_features(self):
        return self._number_of_features

    @property
    def features(self):
        return self._features

    def reset(self):
        self._last_observation = self._env.reset()

    def get_digitized_state_of_last_observation(self):

        digitized_state = list()

        for index, feature in enumerate(self._features):

            digitized_state.append(np.digitize(x=[self.last_observation[index]],
                                               bins=getattr(self, '_'+feature+'_bins'))[0])

        return tuple(digitized_state)
