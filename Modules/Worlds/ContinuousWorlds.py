from Modules.Worlds.BaseWorlds import BaseWorld


class ContinuousCartPoleWorld(BaseWorld):
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
        BaseWorld.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

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


class ContinuousLunarLanderWorld(BaseWorld):
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
        BaseWorld.__init__(self, env_name=env_name, max_episode_steps=max_episode_steps, **kwargs)

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
