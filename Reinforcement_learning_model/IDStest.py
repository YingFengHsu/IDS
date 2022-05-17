import gym
import random
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import numpy as np


class IntrusionDetectionTestEnv(gym.Env):
    def __init__(self):
        self.f = 'en+onehot.csv'
        self.df_xy = np.loadtxt(self.f, delimiter = ",", dtype=np.float32)
        self.df0 = self.df_xy[self.df_xy[:,-1]==0]
        self.df1 = self.df_xy[self.df_xy[:,-1]==1]
        #self.ACTION_TABLE = {1: 'send_notification', -1: 'do_not_send_notificaton'}
        self.ACTION_LOOKUP = {0: 'do_not_send_notificaton', 1: 'send_notification'}

        self.action_space = spaces.Discrete(len(self.ACTION_LOOKUP))
        self.observation_space = spaces.Discrete(self.df_xy.shape[0])
        self.n0, self.n1 = self._get_random_initial_state()
        self.list = np.random.permutation(np.hstack((np.ones(100),np.zeros(0))))
        if self.list[0] == 0:
            self.ob = np.delete(self.df0[self.n0], -1)
            self.n0 += 1
        else:
            self.ob = np.delete(self.df1[self.n1], -1)
            self.n1 += 1
        self.episode_over = False
        self.turns = 0
        #self.sum_rewards = 0.0

    def _step(self, action_index):
        """

        Parameters
        ----------
        action_index :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        #self._take_action(action_index)
        #self.n = self._get_new_state()
        self.reward = self._get_reward(action_index)
        self.turns += 1
        if self.turns < 100:
            if self.list[self.turns] == 0:
                self.ob = np.delete(self.df0[self.n0], -1)
                self.n0 += 1
            else:
                self.ob = np.delete(self.df1[self.n1], -1)
                self.n1 += 1

        if self.turns >= 100:
            self.episode_over = True

        return self.ob, self.reward, self.episode_over, {}

    def _reset(self):
        """
        Reset the environment and supply a new state for initial state
        :return:
        """
        #self.n0, self.n1 = self._get_random_initial_state()
        self.list = np.random.permutation(np.hstack((np.ones(100),np.zeros(0))))
        if self.list[0] == 0:
            self.ob = np.delete(self.df0[self.n0], -1)
            self.n0 += 1
        else:
            self.ob = np.delete(self.df1[self.n1], -1)
            self.n1 += 1
        self.episode_over = False
        self.turns = 0
        #self.sum_rewards = 0.0
        return self.ob


    # def _take_action(self, action_index):
    #     """
    #     Take an action correpsonding to action_index in the current state
    #     :param action_index:
    #     :return:
    #     """
    #     assert action_index < len(self.ACTION_LOOKUP)
    #     action = self.ACTION_LOOKUP[action_index]
    #     # print(action)
    #     return

    def _get_reward(self, action_index):
        """
        Get reward for the action taken in the current state
        :return:
        """
        # df = self.df_xy
        # n = self.n
        y = self.list[self.turns]
        if y == 1.0:
            if action_index == 1:
                reward = 1.0
            else:
                reward = -3.0
        else:
            if action_index == 0:
                reward = 3.0
            else:
                reward = -1.0
        return reward

    # def _get_new_state(self):
    #     """
    #     Get the next state from current state
    #     :return:
    #     """
    #     n = self.n
    #     next_state = n + 1
    #     return next_state

    def _get_random_initial_state(self):
        n0 = 0 #random.randint(0, self.df0.shape[0]-499)
        n1 = 0 #random.randint(0, self.df1.shape[0]-499)
        return n0, n1

    def _seed(self):
        return

    def _render(self, mode='human', close=False):
        return

