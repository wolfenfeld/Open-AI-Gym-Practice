from Worlds import DiscreteCartPoleWorld, CartPoleWorld
from Agents import QLearnerAgent, DQNAgent

import pickle


class Game(object):
    def __init__(self, agents, world, episodes, horizon=None):
        self.agents = agents
        self.world = world
        self.horizon = horizon
        self.episodes = episodes
        self.best_total_reward = 0

    def run(self):
        rewards = list()

        history = dict()

        for episode in range(1, self.episodes + 1):

            total_reward = 0

            state = self.world.reset()

            action = self.agents.sample_action(state)

            for step in range(self.horizon - 1):

                state, reward, done = self.world.interact_with_world(action)

                action = self.agents.sample_action(state)

                total_reward += reward

                if done:
                    reward = -400

                self.agents.update_agent(state, action, reward, episode)

                self.world.last_observation = state

                if done:
                    rewards.append(total_reward)

                    self.best_total_reward = max(total_reward, self.best_total_reward)

                    print('Episode {0} is done.'.format(episode))
                    print('Total Reward : {0}, Best reward : {1}'.format(total_reward, self.best_total_reward))

                    history[episode] = total_reward

                    break

        print('Rewards History for {0}'.format(self.world.env_name))
        print(rewards)
        score_interval = 50
        print('Average reward of last {0} runs'.format(score_interval))
        print(sum(rewards[-score_interval:]) / len(rewards[-score_interval:]))
        return history


class CartPoleGame(Game):

    def __init__(self, world=DiscreteCartPoleWorld(max_episode_steps=200), episodes=200, horizon=200):

        Game.__init__(self,
                      agents=QLearnerAgent(world=world),
                      world=world, episodes=episodes, horizon=horizon)


def run_cart_pole_game(save_data=False, data_file_path=''):
    game = CartPoleGame()
    game.run()

    if save_data:
        pickle.dump(game.agents.qtable, open(data_file_path, 'wb'))


class DQNCartPoleGame(CartPoleGame):

    def __init__(self, world=CartPoleWorld(max_episode_steps=200), episodes=500, horizon=200):

        Game.__init__(self,
                      agents=DQNAgent(world=world),
                      world=world, episodes=episodes, horizon=horizon)


def run_cart_pole_game_dqn():
    game = DQNCartPoleGame()
    history = game.run()
    pickle.dump(history, open('history.pkl', 'wb'))
