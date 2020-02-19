from Modules.Agents import QLearnerAgent, DQNAgent
from Modules.Games.BaseGames import BaseGame
from Modules.Worlds.ContinuousWorlds import CartPoleContinuousWorld, LunarLanderContinuousWorld


class LunarLanderGame(BaseGame):

    def __init__(self, world=LunarLanderContinuousWorld(max_episode_steps=1000), episodes=1000):

        BaseGame.__init__(self, agents=DQNAgent(world=world), world=world, episodes=episodes)
