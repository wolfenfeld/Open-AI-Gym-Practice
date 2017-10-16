from Worlds import CartPoleWorld
from Agents import CartPoleAgent


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

    def __init__(self, world=CartPoleWorld(), episodes=1000, horizon=10000):

        Game.__init__(self,
                      agents=CartPoleAgent(world=world),
                      world=world, episodes=episodes, horizon=horizon)

        self._best_total_reward = 0

    @property
    def best_total_reward(self):
        return self._best_total_reward

    def run(self):
        rewards = list()
        for episode in xrange(self.episodes):

            total_reward = 0

            state = self.world.reset()

            action = self.agents.set_initial_state(state)

            for step in range(self.horizon - 1):

                self.world.interact_with_world(action)

                new_state = self.world.get_digitized_state()
                total_reward += self.world.last_reward

                if self.world.is_done:
                    reward = -200
                else:
                    reward = self.world.last_reward

                action = self.agents.act(new_state=new_state,
                                         last_reward=reward)
                if self.world.is_done:
                    rewards.append(total_reward)
                    if total_reward > self.best_total_reward:
                        self._best_total_reward = total_reward
                    print 'Episode {0} is done.'.format(episode)
                    print 'Total Reward : {0}, Best reward : {1}'.format(total_reward, self.best_total_reward)
                    break

        print rewards

if __name__ == "__main__":
    game = CartPoleGame()
    game.run()