import datetime as dt
import math
import random
from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from tqdm import tqdm


class AI_Trader():

    def __init__(self, state_size, action_space=3, model_name="AITrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.model_name = model_name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.model = self.model_builder()

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(units=32, activation="relu"))
        model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        model.add(tf.keras.layers.Dense(
            units=self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001))
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)

        actions = self.model.predict(state[0])
        return np.argmax(actions[0])

    def batch_train(self, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            if not done:
                reward = reward + self.gamma * \
                    np.amax(self.model.predict(next_state[0]))

            target = self.model.predict(state[0])
            target[0][action] = reward

            self.model.fit(state[0], target, epochs=1,
                           verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def stocks_price_format(n):
    if (n < 0):
        return '- $ {0:2f}'.format(abs(n))
    else:
        return '+ $ {0:2f}'.format(abs(n))


def stocker_market(ticker='AAPL', period='max', interval='1mo'):
    dataset = yf.download(ticker, period=period, interval=interval)
    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]
    close_stocker = dataset['Close']

    return close_stocker


def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1

    if starting_id >= 0:
        windowed_data = np.array(data[starting_id:timestep + 1])
    else:
        windowed_data = np.array(- starting_id *
                                 [data[0]] + list(data[0:timestep + 1]))

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

    return np.array([state]), windowed_data


# PARAMS
episodes = 1000
batch_size = 32

ticker = 'AAPL'
period = 'max'
interval = '1mo'

timestep = 0
window_size = 10

# Exemplo de uso
data = stocker_market(ticker=ticker,
                      period=period, interval=interval)

s, w = state_creator(data, timestep, 5)


trader = AI_Trader(window_size)
trader.model.summary()

for episode in range(episodes + 1):
    print("Episode: {}/{}".format(episode, episodes))
    state = state_creator(data,  0, window_size + 1)

    total_profit = 0
    trader.inventory = []

    for t in tqdm(range(len(data-1))):
        action = trader.trade(state)
        next_state = state_creator(data, t+1, window_size + 1)

        reward = 0

        if action == 1:  # buying stocker
            trader.inventory.append(data[t])
            print("AI Trader bought: ", stocks_price_format(data[t]))
        elif action == 2 and len(trader.inventory) > 0:  # selling stocker
            buy_price = trader.inventory.pop(0)

            reward = max(data[t] - buy_price, 0)
            total_profit += data[t] - buy_price
            print("AI Trader sold: ", stocks_price_format(
                data[t]), " Profit: " + stocks_price_format(data[t] - buy_price))

        if (t == len(data-1) - 1):
            done = True
        else:
            done = False

        trader.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print("##########################")
            print("Total Profit: {}".format(total_profit))
            print("##########################")
        if len(trader.memory) > batch_size:
            trader.batch_train(batch_size)
    if episode % 10:
        trader.model.save("ai_trader_{}.h5".format(episode))

print("END")
