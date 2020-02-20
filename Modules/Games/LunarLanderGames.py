from Modules.Agents.OffPolicyAgents import DQNAgent
from Modules.Games.BaseGames import BaseGame
from Modules.Worlds.ContinuousWorlds import ContinuousLunarLanderWorld


class LunarLanderGame(BaseGame):

    def __init__(self, world=ContinuousLunarLanderWorld(max_episode_steps=1000), episodes=1000):

        BaseGame.__init__(self, agent=DQNAgent(world=world), world=world, episodes=episodes)
