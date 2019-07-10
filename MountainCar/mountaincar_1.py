# import plaidml.keras
# plaidml.keras.install_backend()
import gym
from gym import wrappers, logger
import keras
from keras import Sequential
from keras.layers import Activation, Dense
from collections import deque
from keras.models import save_model, load_model
import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# print(sys.path)

render_env = 0
batch_size = 32
memory_len = 20000

save_name = 'saved_model.h5'



class Agent(object):
    def __init__(self, env, memory_len, batch_size):
        self.env = env
        self.action_space = env.action_space
        self.input_shape = env.observation_space.shape  # box space: shape = (2,)   (position, velocity)
        self.reshape_state = self.input_shape[0]
        self.action_shape = env.action_space.n  # discrete space: n = 3
        # create model for learning q values
        self.queue = deque(maxlen=memory_len)
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.05
        self.gamma = 1
        self.learning_rate = 0.001

        self.model_states = self.create_dnn()
        self.model_targets = self.create_dnn()

    def act(self, state):
        # return self.action_space.sample()
        # while observation has a shape pf (2,) it needs to be reshaped since keras wants each input
        # as a single (2,)
        if np.random.rand(0, 1) < self.epsilon:
            return self.action_space.sample()

        return np.argmax(self.model_states.predict(x=state)[0])

    def create_dnn(self):
        model = Sequential([
            Dense(units=24, input_shape=self.input_shape),
            Activation('relu'),
            Dense(units=48),
            Activation('relu'),
            Dense(units=self.action_shape),
            Activation('linear')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(lr=self.learning_rate),
            loss='mse'
        )
        return model

    def save_state(self, state, action, next_state, reward, done):
        self.queue.append([state, action, next_state,  reward, done])

    def update_dnn(self):
        """
        x_batch, y_batch = [], []
        minibatch = random.sample(self.queue, min(len(self.queue), batch_size))
        for state, action, next_state, reward, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward +\
                self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        """

        if len(self.queue) > batch_size:
            """
            queue_sample = random.sample(self.queue, batch_size)
            states = np.array([q[0] for q in queue_sample])
            actions = np.array([q[1] for q in queue_sample])
            rewards = np.array([q[3] for q in queue_sample])
            next_states = np.array([q[2] for q in queue_sample])
            done = np.array([q[4] for q in queue_sample])

            states = np.array(states).reshape(batch_size, self.reshape_state)
            next_states = np.array(next_states).reshape(batch_size, self.reshape_state)

            state_targets = self.model_states.predict(states)
            next_state_targets = self.model_targets.predict(next_states)

            for i in range(0, batch_size):
                target = state_targets[i]
                target[actions[i]] = rewards[i] + self.gamma * max(next_state_targets[i]) if not done[i] else rewards[i]

            self.model_states.fit(states, state_targets, epochs=1, verbose=False)
            """
            states_arr, next_states_arr, action_arr, rewards_arr, done_arr = [], [], [], [], []
            minibatch = random.sample(self.queue, batch_size)
            for state, action, next_state, reward, done in minibatch:
                states_arr.append(state)
                next_states_arr.append(next_state)
                action_arr.append(action)
                rewards_arr.append(reward)
                done_arr.append(done)

            states = np.array(states_arr).reshape(batch_size, self.reshape_state)
            next_states = np.array(next_states_arr).reshape(batch_size, self.reshape_state)
            rewards = np.array(rewards_arr).reshape(batch_size, 1)
            dones = np.array(done_arr).reshape(batch_size, 1)
            actions = np.array(action_arr).reshape(batch_size, 1)

            state_targets = self.model_states.predict(states)
            next_state_targets = self.model_targets.predict(next_states)
            for i in range(0, batch_size):
                target = state_targets[i]
                target[actions[i]] = rewards[i] +\
                    self.gamma * max(next_state_targets[i]) if not dones[i] else rewards[i]

            self.model_states.fit(states, state_targets, epochs=1, verbose=0)

            """
            target = self.model.predict(states[i])[0]
            target = reshape_input(target)
            print(target)
            next_q_state = self.model.predict(next_states[i])[0]
            reshape_input(next_q_state)
            print(next_q_state)
            max_q = max(next_q_state)
            print(target, actions, rewards, done)
            target[actions[i]] = (rewards[i] + (0.98 * max_q if not done[i] else 0))
            self.model.fit(states[i], target)
            """

    def update_target_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        self.model_targets.set_weights(self.model_states.get_weights())

    def run(self, run_threshold, run_from_save):
        # logger.set_level(logger.INFO)
        global render_env
        all_rewards, heights = [], []
        # train network to optimal q_values
        i = 0
        complete = 0    # false until victory has been achieved 5 times in a row

        if run_from_save:
            load_model(save_name, compile=True)
            self.model_targets = load_model(save_name, compile=True)

        while complete < run_threshold:
            ob = self.env.reset()
            next_state = np.array(ob).reshape(1, self.reshape_state)
            reward_sum = 0
            max_height = -9999
            t = 0       # duration of the episode
            while t < 200:
                state = next_state
                action = agent.act(state)
                ob, reward, done, _ = self.env.step(action)
                next_state = np.array(next_state).reshape(1, self.reshape_state)

                if next_state[0][0] >= 0.5: # reward for reaching the top
                    reward += 10

                if next_state[0][0] > max_height:
                    max_height = next_state[0][0]

                reward_sum += reward
                if not run_from_save:
                    agent.save_state(state, action, next_state, reward, done)
                    agent.update_dnn()
                if not i % 20 or run_from_save:
                    self.env.render()
                t += 1
                if done:

                    if t < 200:
                        print(f"victory with reward : {reward_sum} on episode {i}")
                        complete += 1
                    else:
                        complete = 0
                    if not i % 20:
                        print(f"episode {i}")

                    all_rewards.append(reward_sum)
                    heights.append(max_height)
                    if not run_from_save:
                        self.update_target_model()
                    i += 1
                    break
        # Close the env and write monitor result info to disk

        try:
            print(f"hill has been climbed {run_threshold} times in a row. this took {i} episodes")
            self.env.close()
            del self.env
            if not run_from_save:
                save_model(agent.model_states, save_name,include_optimizer=True)
            plt.plot([i for i in range(0, i)], all_rewards)
            plt.show()
            plt.plot([i for i in range(0, i)], heights)
            plt.show()
        except ImportError:
            pass


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.seed(0)
    outdir = './tmp/random-agent-results-0'
    env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    agent = Agent(env, memory_len=memory_len, batch_size=batch_size)
    agent.run(run_threshold=2, run_from_save=1)

