from Worlds import CartPoleWorld
from Agents import CartPoleAgent

import pickle


class Game(object):
    def __init__(self, agents, world, episodes, horizon=None):
        self._agents = agents
        self._world = world
        self._horizon = horizon
        self._episodes = episodes

    @property
    def agents(self):
        return self._agents

    @property
    def world(self):
        return self._world

    @property
    def horizon(self):
        return self._horizon

    @property
    def episodes(self):
        return self._episodes


class CartPoleGame(Game):

    def __init__(self, world=CartPoleWorld(max_episode_steps=500), episodes=1000, horizon=500):

        Game.__init__(self,
                      agents=CartPoleAgent(world=world),
                      world=world, episodes=episodes, horizon=horizon)

        self._best_total_reward = 0

    @property
    def best_total_reward(self):
        return self._best_total_reward

    def run(self):
        rewards = list()
        for episode in range(self.episodes):

            total_reward = 0

            self.world.reset()

            self.agents.set_initial_state(self.world.get_digitized_state_of_last_observation())

            for step in range(self.horizon - 1):

                if episode == self.episodes - 2:
                    self.world.render()

                self.world.interact_with_world(self.agents.last_action)

                self.world.get_digitized_state_of_last_observation()
                total_reward += self.world.last_reward

                if self.world.is_done:
                    reward = -200
                else:
                    reward = self.world.last_reward

                self.agents.act(new_state=self.world.get_digitized_state_of_last_observation(),
                                last_reward=reward)

                if self.world.is_done:

                    rewards.append(total_reward)

                    self._best_total_reward = max(total_reward, self.best_total_reward)

                    print 'Episode {0} is done.'.format(episode)
                    print 'Total Reward : {0}, Best reward : {1}'.format(total_reward, self.best_total_reward)
                    break

        print rewards


def run_cart_pole_game(save_data=False, data_file_path=''):
    game = CartPoleGame()
    game.run()

    if save_data:
        pickle.dump(game.agents.qtable, open(data_file_path, 'wb'))
