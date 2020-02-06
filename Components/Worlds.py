import numpy as np

import gym

class World(object):
    """
    The world object where the game is played.
    """
    def __init__(self, env_name='', max_episode_steps=200, **kwargs):
        """
        Initializing the world object.
        :param env_name: name of the environment.
        :param max_episode_steps: maximum episodes in a single game.
        :param kwargs: rest of the arguments/parameters.
        """
        self.env_name = env_name
        # Creating the environment.
        self.env = gym.make(self.env_name)
        # Overwriting the number of max episodes that can be played in a single episode.
        self.env._max_episode_steps = max_episode_steps
        # The last observation
        self.last_observation = None
        # The number of steps that can be sampled.
        self.number_of_actions = self.env.action_space

        for k, v in kwargs.items():
            setattr(self, '_'+k, v)

    def reset(self):
        """
        Reseting the world object.
        :return: returning the initial state
        """
        return self.env.reset()

    def step(self, action):
        """
        Taking a step with an action.
        :param action: the taken action.
        :return: returning the observation,
        """
        return self.env.step(action)

    def render(self):
        """
        Render the current environment.
        """
        self.env.render()

    def interact_with_world(self, action):
        """
        performing an action on the environment.
        :param action: the action that is performed on the environment.
        """
        pass


class DiscreteWorld(World):
    """
    A discrete world object - the states will be indexed.
    """
    def __init__(self,
                 env_name,
                 max_episode_steps=200,
                 number_of_discrete_values_per_feature=10,
                 **kwargs):
        """
        Initializing the discrete world object
        :param env_name: name of the environment.
        :param max_episode_steps: maximum number of steps in any episode of the game.
        :param number_of_discrete_values_per_feature: number different values of a feature in the environment.
        :param kwargs: rest of the parameters/arguments.
        """
        World.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        self.number_of_discrete_values_per_feature = number_of_discrete_values_per_feature


class DiscreteCartPoleWorld(DiscreteWorld):
    """
    Discrete cart-pole world object.
    """
    def __init__(self,
                 env_name='CartPole-v0',
                 max_episode_steps=200,
                 **kwargs):
        """
        Initializing the cart-pole world object.
        :param env_name: name of the environment.
        :param max_episode_steps: maximum number of steps in an episode of the game.
        :param kwargs: rest of the parameters/arguments.
        """
        DiscreteWorld.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        # Setting the number of action for this world.
        self.number_of_actions = self.env.action_space.n

        # The features of the cart-pole world.
        self.features = ['cart_position', 'pole_angle', 'cart_velocity', 'angle_rate']
        self.number_of_features = self.env.observation_space.shape[0]

        # Indexing all the possible states of the world.
        self.cart_position_bins = np.linspace(start=-3, stop=3, num=self.number_of_discrete_values_per_feature)
        self.pole_angle_bins = np.linspace(start=-5, stop=5, num=self.number_of_discrete_values_per_feature)
        self.cart_velocity_bins = np.linspace(start=-1, stop=1, num=self.number_of_discrete_values_per_feature)
        self.angle_rate_bins = np.linspace(start=-5, stop=5, num=self.number_of_discrete_values_per_feature)

    def reset(self):
        """
        Reseting the world.
        :return: the initial state (digitized)
        """

        self.last_observation = self.digitize_step(self.env.reset())

        return self.last_observation

    def digitize_step(self, state):
        """
        Digitizing a state (according to the number_of_discrete_values_per_feature).
        :param state: the state that will be digitized
        :return: the digitized state as a tuple.
        """
        digitized_state = list()

        for index, feature in enumerate(self.features):
            digitized_state.append(np.digitize(x=[state[index]],
                                               bins=getattr(self, feature + '_bins'))[0])

        return tuple(digitized_state)

    def interact_with_world(self, action):
        """
        Playing the action on the world and receiving the feedback.
        :param action: that taken action.
        :return: the feedback, where the state is digitized.
        """

        state, reward, done, _ = self.step(action)

        return self.digitize_step(state), reward, done


class CartPoleWorld(World):
    """
    A continues (not digitized) cart-pole world.
    """
    def __init__(self,
                 env_name='CartPole-v0',
                 max_episode_steps=200,
                 **kwargs):
        """
        Initializing the cart-pole world object
        :param env_name: name of the environment.
        :param max_episode_steps: maximum steps that can be taken during an episode in the game.
        :param kwargs:  rest of the parameters/arguments.
        """
        World.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        # Number of action that can be taken in a state.
        self.number_of_actions = self.env.action_space.n

        # The features of this world.
        self.features = ['cart_position', 'pole_angle', 'cart_velocity', 'angle_rate']
        self.number_of_features = self.env.observation_space.shape[0]

    def reset(self):
        """
        Resting the world.
        :return: the initial state.
        """
        self.last_observation = self.env.reset()

        return self.last_observation

    def interact_with_world(self, action):
        """
        Playing the action on the world.
        :param action: the taken action
        :return: the feedback from the world.
        """

        state, reward, done, _ = self.step(action)

        return state, reward, done


class LunarLanderWorld(World):
    """
    A continues (not digitized) lunar-lander world.
    """
    def __init__(self,
                 env_name='LunarLander-v2',
                 max_episode_steps=1000,
                 **kwargs):
        """
        Initializing the cart-pole world object
        :param env_name: name of the environment.
        :param max_episode_steps: maximum steps that can be taken during an episode in the game.
        :param kwargs:  rest of the parameters/arguments.
        """
        World.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        # Number of action that can be taken in a state.
        self.number_of_actions = self.env.action_space.n

        # The features of this world.
        self.number_of_features = self.env.observation_space.shape[0]

    def reset(self):
        """
        Resting the world.
        :return: the initial state.
        """
        self.last_observation = self.env.reset()

        return self.last_observation

    def interact_with_world(self, action):
        """
        Playing the action on the world.
        :param action: the taken action
        :return: the feedback from the world.
        """

        state, reward, done, _ = self.step(action)

        return state, reward, done

