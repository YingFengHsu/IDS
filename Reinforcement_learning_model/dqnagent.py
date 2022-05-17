import numpy as np
import gym
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy, EpsGreedyQPolicy, BoltzmannQPolicy, MaxBoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


ENV_NAME = 'OUIDS-v0'

env = gym.make(ENV_NAME)
env.reset()
envt = gym.make('IDStest-v0')
envt.reset()
nb_actions = 2

model = Sequential()
model.add(Flatten(input_shape=(1,) + (320,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

memory = SequentialMemory(limit=20000, window_length=1)
policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-4), metrics = ['mae'])
hist = dqn.fit(env, nb_steps = 2100000, visualize = True, verbose=2)

# results = {}
# test_episodes = 172
# t1 = time.time()
# test = dqn.test(envt, nb_episodes=test_episodes, visualize=True)
# print(test_episodes*100/(time.time()-t1))
# #results[steps] = np.mean(test.history['episode_reward'])
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(range(1100), [(s+200)/4 for s in hist.history['episode_reward']])
# plt.ylim(50,100)
# fig.savefig('acc_per_episode.png')
#
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1,1)
# # ax.plot(range(1,31), [(s+200)/4 for s in test.history['episode_reward']])
# # plt.ylim(50,100)
# # fig.savefig('acc_per_episode_test.png')
#
# print(np.sum([(s+300)/4 for s in test.history['episode_reward']]))
# print(np.mean([(s+100)/4 for s in test.history['episode_reward']]))
