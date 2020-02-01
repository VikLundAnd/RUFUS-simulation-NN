from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add


class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.005
        self.model = self.network()
        #self.model = self.network('weights.hdf5')
        self.epsilon = 0
        self.actual = []
        self.memory = []

    def get_state(self, robot):

        #State with either 1 or 0.
#        state = [
#            (((robot.mbody1.GetRot()).Q_to_Rotv()).z < 3.25), #lean forward
#            (((robot.mbody1.GetRot()).Q_to_Rotv()).z > 3.25), #lean backwards
#            ]
#
        #Convert to 1 and 0
#        for i in range(len(state)):
#            if state[i]:
#                state[i]=1
#            else:
#                state[i]=0


        #State with gradient from 1 to 0.
        state = [
            -(((robot.mbody1.GetRot()).Q_to_Rotv()).z - 3.25), #lean forward
            (((robot.mbody1.GetRot()).Q_to_Rotv()).z - 3.25), #lean backwards
            ]

        for i in range(len(state)):
            if state[i] > 1 or state[i] < 0:
                state[i]=0

        print(state)
        return np.asarray(state)

    def set_reward(self, score, highscore, previousScore, amountOfSimulations, highscoreTime):
        #self.reward = (score - highscore) * (amountOfSimulations - highscoreTime)
        self.reward = (score - highscore)
        #self.reward += score - previousScore
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=2))
        model.add(Dropout(0.15))
        #model.add(Dense(output_dim=120, activation='relu'))
        #model.add(Dropout(0.15))
        model.add(Dense(output_dim=60, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=2, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, keepRunning):
        self.memory.append((state, action, reward, next_state, keepRunning))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, keeprunning):
        target = reward
        if keeprunning:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 2)))[0])
        target_f = self.model.predict(state.reshape((1, 2)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 2)), target_f, epochs=1, verbose=0)
        self.model.save_weights('weights.hdf5')
