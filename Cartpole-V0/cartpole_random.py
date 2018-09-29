import gym
import numpy as np

env = gym.make('CartPole-v0')
episode_count, iterationscount = 100, 50
rewardstore = []
iterations = np.empty(iterationscount)
maxreward = 0


def reset():
    global maxreward, model
    if maxreward < 200 or maxreward == 0:
        model = np.random.rand(4) * 2 - 1  # model initialised to random weights between -1 and 1
    maxreward = 0


def train():
    global model, maxreward, rewardstore
    for i_episode in range(0, episode_count):
        observation, done, episode_reward = env.reset(), False, 0
        test_model = np.random.rand(4) * 2 - 1
        while not done:
            action = random_attempts(observation, test_model)
            observation, reward, done, info = env.step(action)
            episode_reward += reward

        if episode_reward > maxreward:
            maxreward = episode_reward
            rewardstore.append(maxreward)
            model = test_model
            if maxreward == 200:
                break
    return i_episode


def random_attempts(obs, test_model):
    prod = test_model @ obs
    return 0 if prod < 0 else 1


def demonstrate():
    while True:
        observation, done = env.reset(), False
        while not done:
            env.render()
            action = random_attempts(observation, model)
            observation, reward, done, info = env.step(action)


for x in range(0, iterationscount):
    reset()
    iterations[x] = train()

print("average iterations", np.average(iterations))
print("values are", model)
demonstrate()


env.close()  # to avoid error when closing file
try:
    del env
except ImportError:
    pass
