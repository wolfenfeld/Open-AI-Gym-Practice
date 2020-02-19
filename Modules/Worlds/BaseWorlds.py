import gym


class BaseWorld(object):
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
        Resetting the world object.
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
