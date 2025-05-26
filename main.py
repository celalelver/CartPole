import numpy as np
import pandas as pd
import matplotlib.pyplot
import gym
from collections import deque
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random
import time

class DQLAgent:
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation="tanh"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state, verbose=0)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            train_target = self.model.predict(state, verbose=0)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)

    def adaptiveEGready(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        pass


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = DQLAgent(env)
    batch_size = 16
    episodes = 100
    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, 4])
        time_step = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)

            agent.adaptiveEGready()
            time_step += 1

            if done:
                print("Episode : {} , time : {} ".format(e, time_step))
                break

print("\nTraining is complete. Model is being tested...\n")

trained_model = agent
state, _ = env.reset()
state = np.reshape(state, [1, 4])
time_t = 0
while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _, _ = env.step(action)
    next_state = np.reshape(next_state, [1, 4])
    state = next_state
    time_t += 1
    print(f"Adım: {time_t}")
    time.sleep(0.02)
    if done:
        break
env.close()
print("Done")
