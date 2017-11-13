from Worlds import DiscreteCartPoleWorld, CartPoleWorld, LunarLanderWorld
from Agents import QLearnerAgent, DQNAgent

import pickle


class Game(object):
    """
    Game - The object that defines the game.
    """
    def __init__(self, agents, world, episodes):
        """
        Initialization of the Game object
        :param agents: the agents that play the game
        :param world: the world where the game is played
        :param episodes: the number of episodes that will be played in the game.
        """
        self.agents = agents
        self.world = world
        self.episodes = episodes

        # The best total reward over all the episodes.
        self.best_total_reward = 0

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
            action = self.agents.sample_action(state)

            while True:
                # if episode > self.episodes - 300:
                self.world.render()

                # Interacting with the world and acquiring the feedback:
                # the new state, the reward and the done indicator.
                state, reward, done = self.world.interact_with_world(action)

                # Sampling the action from the agent.
                action = self.agents.sample_action(state)

                # Updating the total reward.
                total_reward += reward

                # Updating the agent
                self.agents.update_agent(state, action, reward, episode, done)

                # Updating the world object.
                self.world.last_observation = state

                # If the episode is done.
                if done:

                    # Updating the reward dictionary.
                    rewards[episode] = total_reward

                    # Keeping best reward.
                    self.best_total_reward = max(total_reward, self.best_total_reward)

                    print('Episode {0} is done.'.format(episode))
                    print('Total Reward : {0}, Best reward : {1}'.format(total_reward, self.best_total_reward))

                    break

        print('Rewards History for {0}'.format(self.world.env_name))
        print(rewards.values())
        score_interval = 50
        print('Average reward of last {0} runs'.format(score_interval))
        print(sum(rewards.values()[-score_interval:]) / len(rewards.values()[-score_interval:]))
        return rewards


class CartPoleGame(Game):

    def __init__(self, world=DiscreteCartPoleWorld(max_episode_steps=400), episodes=600):

        Game.__init__(self, agents=QLearnerAgent(world=world), world=world, episodes=episodes)


def run_cart_pole_game(save_data=False, data_file_path=''):
    game = CartPoleGame()
    game.run()

    if save_data:
        pickle.dump(game.agents.qtable, open(data_file_path, 'wb'))


class DQNCartPoleGame(CartPoleGame):

    def __init__(self, world=CartPoleWorld(max_episode_steps=300), episodes=600):

        Game.__init__(self, agents=DQNAgent(world=world), world=world, episodes=episodes)


def run_cart_pole_game_dqn():
    game = DQNCartPoleGame()
    history = game.run()
    print(len(history.keys()))
    pickle.dump(history, open('history.pkl', 'wb'))


class LunarLanderGame(Game):

    def __init__(self, world=LunarLanderWorld(max_episode_steps=1000), episodes=1000):

        Game.__init__(self, agents=DQNAgent(world=world), world=world, episodes=episodes)


def run_lunar_lander_game(save_data=False, data_file_path=''):
    game = LunarLanderGame()
    game.run()

    if save_data:
        pickle.dump(game.agents.qtable, open(data_file_path, 'wb'))
