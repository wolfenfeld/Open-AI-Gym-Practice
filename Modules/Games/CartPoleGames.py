from Modules.Agents.OffPolicyAgents import QLearnerAgent, DQNAgent, HillClimbAgent
from Modules.Games.BaseGames import BaseGame
from Modules.Worlds.ContinuousWorlds import ContinuousCartPoleWorld
from Modules.Worlds.DescreteWorlds import DiscreteCartPoleWorld
from Modules.DecisionModels.BaseDecisionModel import Transition


class CartPoleGame(BaseGame):

    def __init__(self, world=DiscreteCartPoleWorld(max_episode_steps=400), episodes=600):

        BaseGame.__init__(self, agent=QLearnerAgent(world=world), world=world, episodes=episodes)


class HillClimbCartPoleGame(BaseGame):

    def __init__(self, world=DiscreteCartPoleWorld(max_episode_steps=400), episodes=600):

        BaseGame.__init__(self, agent=HillClimbAgent(world=world), world=world, episodes=episodes)

    def run(self):
        """
        run function - Running a game
        """
        # The total rewards of each episode.
        rewards = dict()

        for episode in range(1, self.episodes + 1):

            # The total reward for the current episode.
            total_reward = 0

            # The initial state
            state = self.world.reset()

            # The initial action
            action = self.agent.get_action(state)

            while True:
                # if episode > self.episodes - 300:
                self.world.render()

                # Interacting with the world and acquiring the feedback:
                # the new state, the reward and the done indicator.
                state, reward, done = self.world.interact_with_world(action)

                # Sampling the action from the agent.
                action = self.agent.get_action(state)

                # Updating the total reward.
                total_reward += reward

                # Updating the world object.
                self.world.last_observation = state

                # If the episode is done.
                if done:

                    # Updating the reward dictionary.
                    rewards[episode] = total_reward

                    transition = Transition(None, None, total_reward, None, None)
                    # Updating the agent
                    self.agent.reinforce(episode, transition)

                    # Keeping best reward.
                    self.best_total_reward = max(total_reward, self.best_total_reward)

                    print('Episode {0} is done.'.format(episode))
                    print('Total Reward : {0}, Best reward : {1}'.format(total_reward, self.best_total_reward))

                    break

        print('Rewards History for {0}'.format(self.world.env_name))
        print(rewards.values())
        score_interval = 50
        print('Average reward of last {0} runs'.format(score_interval))
        print(sum(list(rewards.values())[-score_interval:]) / len(list(rewards.values())[-score_interval:]))

        return rewards


class DQNCartPoleGame(CartPoleGame):

    def __init__(self, world=ContinuousCartPoleWorld(max_episode_steps=300), episodes=600):

        BaseGame.__init__(self, agent=DQNAgent(world=world), world=world, episodes=episodes)