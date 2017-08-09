import gym
import numpy as np



def run_episode(env, parameters):

    #Running an episode

    observation = env.reset()
    total_reward = 0

    # For 200 time-steps
    for t in xrange(200):

        env.render()

        # Initialize random weights
        if np.matmul(parameters, observation) < 0:
            action = 0
        else:
            action = 1

        observation, reward, done, info = env.step(action)
        total_reward += reward

        if done:
            break

    return total_reward


# Hill climbing algorithm training
def train():
    env = gym.make('CartPole-v0')

    noise_scaling = 0.1
    parameters = np.random.rand(4) * 2 - 1
    best_reward = 0

    # 2000 episodes
    for t in xrange(100):

        new_params = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
        reward = run_episode(env, new_params)

        print "reward %d best %d" % (reward, best_reward)
        if reward > best_reward:
            best_reward = reward
            parameters = new_params

            if reward == 200:
                break

    return best_reward


if __name__ == "__main__":
    r = train()
    print r
