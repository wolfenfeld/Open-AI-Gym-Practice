from Modules.Agents.OffPolicyAgents import QLearnerAgent, DQNAgent
from Modules.Games.BaseGames import BaseGame
from Modules.Worlds.ContinuousWorlds import ContinuousCartPoleWorld
from Modules.Worlds.DescreteWorlds import DiscreteCartPoleWorld


class CartPoleGame(BaseGame):

    def __init__(self, world=DiscreteCartPoleWorld(max_episode_steps=400), episodes=600):

        BaseGame.__init__(self, agents=QLearnerAgent(world=world), world=world, episodes=episodes)


class DQNCartPoleGame(CartPoleGame):

    def __init__(self, world=ContinuousCartPoleWorld(max_episode_steps=300), episodes=600):

        BaseGame.__init__(self, agents=DQNAgent(world=world), world=world, episodes=episodes)