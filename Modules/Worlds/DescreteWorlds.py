import numpy as np

from Modules.Worlds.BaseWorlds import BaseWorld


class DiscreteBaseWorld(BaseWorld):
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
        BaseWorld.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

        self.number_of_discrete_values_per_feature = number_of_discrete_values_per_feature

    def interact_with_world(self, action):
        """
        performing an action on the environment.
        :param action: the action that is performed on the environment.
        """
        raise NotImplementedError


class DiscreteCartPoleWorld(DiscreteBaseWorld):
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
        DiscreteBaseWorld.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

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
        Resetting the world.
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
