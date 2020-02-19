class BaseDecisionModel(object):
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
        raise NotImplementedError

    def get_random_action(self):
        """
        Returning a random action.
        :return: a random action.
        """
        raise NotImplementedError
