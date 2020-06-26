import gym
from gym import spaces
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import random
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Activation
from tensorflow.keras import Input
from tensorflow import convert_to_tensor as convert
from collections import deque
import pickle
path_base = ''

def softmax(y, theta = 1.0):
    y = y * float(theta)
    y = y - np.max(y)
    y = np.exp(y)
    ax_sum = np.sum(y)
    p = y / ax_sum
    return p*0.9

class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, params, train = True):
        super(StockEnv,self).__init__()
        self.initbalance = params['balance']
        self.initshares_held = params['shares_held']
        self.num_stocks = params['num_stocks']
        self.balance_normal = params['balance_normal']
        self.price_normal = params['price_normal']
        self.num_prev = params['num_prev']
        self.shares_normal = params['shares_normal']
        self.dfs = df
        self.state_dimensions = self.num_prev*self.num_stocks*4+self.num_stocks+1
        self.train = train

        assert len(df) == self.num_stocks, "Size of database not equal to number of stocks"

        self.max_steps = min([len(d.loc[:,'Open']) for d in self.dfs])
        self.action_space = spaces.Box(low = np.zeros(self.num_stocks*2), high = np.ones(self.num_stocks*2), dtype = np.float32)
        self.observation_space = spaces.Box(low = -np.ones(self.state_dimensions), high = np.ones(self.state_dimensions), dtype = np.float32)

    def reset(self):
        self.balance = self.initbalance
        if self.train:
            self.current_step = np.random.randint(self.num_prev, self.max_steps)
        else:
            self.current_step = self.num_prev
        self.shares_held = self.initshares_held
        if self.shares_held is None:
            self.shares_held = np.zeros((1, self.num_stocks))
        self.current_price = self.get_price()
        self.highest_price = 0
        self.net_worth = self.balance + np.sum(self.shares_held*self.current_price)
        self.initial_worth = self.net_worth
        self.max_net_worth = self.net_worth
        self.set_high()
        self.done = False
        self.frame = np.zeros((1, self.state_dimensions))
        self.info = {
            'current_step' : self.current_step,
            'current_price': self.current_price,
            'highest_price': self.highest_price,
            'net_worth' : self.net_worth,
            'max_net_worth': self.max_net_worth,
            'shares_held' : self.shares_held,
            'shares_normal' : self.shares_normal,
            'balance_normal' : self.balance_normal,
            'balance' : self.balance,
        }
        ons,_ =  self.observe()
        return ons
        
    def get_price(self):
        return np.array([np.random.uniform(df.loc[self.current_step,"Low"], df.loc[self.current_step,"High"]) for df in self.dfs]).reshape((1, self.num_stocks))
      
    def set_high(self):
        high = np.array([df.loc[self.current_step, 'High'] for df in self.dfs]).reshape((1, self.num_stocks))
        self.highest_price = np.maximum(self.highest_price, high)

    def get_state(self,i):
        return [self.dfs[i].loc[self.current_step-self.num_prev+2:self.current_step+1,'Open'],self.dfs[i].loc[self.current_step-self.num_prev+2:self.current_step+1,'High'],self.dfs[i].loc[self.current_step-self.num_prev+2:self.current_step+1,'Low'],self.dfs[i].loc[self.current_step-self.num_prev+2:self.current_step+1,'Close']]
    
    def observe(self):
#         print(self.current_step)
        frame = self.frame.copy()
        for i in range(self.num_stocks):
            frame[0, 4*self.num_prev*i:self.num_prev*4*i+self.num_prev*4] = np.array(self.get_state(i)).reshape((1,4*self.num_prev)) - self.frame[0, 4*self.num_prev*i:self.num_prev*4*i+self.num_prev*4]
            self.frame[0, 4*self.num_prev*i:self.num_prev*4*i+self.num_prev*4] = self.frame[0, 4*self.num_prev*i:self.num_prev*4*i+self.num_prev*4] + frame[0, 4*self.num_prev*i:self.num_prev*4*i+self.num_prev*4]
            frame[0, 4*self.num_prev*i:self.num_prev*4*i+self.num_prev*4] /= self.price_normal
            
        frame[0, self.num_prev*self.num_stocks*4:-1] = self.frame[0, self.num_prev*self.num_stocks*4:-1] = self.shares_held/self.shares_normal
        frame[0, -1] =  self.frame[0, -1] = self.balance/self.balance_normal
        self.info = {
            'current_step' : self.current_step,
            'current_price': self.current_price,
            'highest_price': self.highest_price,
            'net_worth' : self.net_worth,
            'max_net_worth': self.max_net_worth,
            'shares_held' : self.shares_held,
            'shares_normal' : self.shares_normal,
            'balance_normal' : self.balance_normal,
            'balance' : self.balance
        }
        return frame.reshape(self.state_dimensions), self.info
    
    def update_worth(self, reward):
        self.net_worth += reward
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

    def update_balance(self, action):
        self.balance += np.sum(action[:, :self.num_stocks]*self.current_price)
        self.balance -= np.sum(action[:, self.num_stocks:]*self.current_price)

    def update_shares(self, action):
        self.shares_held -= action[:, :self.num_stocks]
        self.shares_held +=  action[:, self.num_stocks:]

    def take_action(self, action):
        self.current_price = self.get_price()
        action[:, :self.num_stocks] = np.floor(action[:, :self.num_stocks] *self.shares_held)
        action[:, self.num_stocks:] = np.floor(action[:, self.num_stocks:] * self.balance / self.current_price)
        self.set_high()
        self.update_balance(action)
        self.update_shares(action)
        reward = self.balance + np.sum(self.shares_held * self.current_price) - self.net_worth
        self.update_worth(reward)
        return reward
            
    def step(self, actions):
        buy_sum = np.sum(actions[self.num_stocks:])
        actions[self.num_stocks:] = softmax(actions[self.num_stocks:])
        actions[self.num_stocks:] *= np.min([buy_sum,1])
#         print(actions)
        action = actions.reshape((1,self.num_stocks*2)).copy()
        self.current_step += 1
        if self.current_step >= self.max_steps-1 or self.done:
            self.done = True
            return np.zeros((self.state_dimensions)), 0, self.done, self.info
        if np.sum(action[:, self.num_stocks:]) > 1:
            print(action)
            print('gadbad')
        reward = self.take_action(action)
        self.done = self.net_worth <= self.initial_worth*0.05
        if self.done:
            print('snap')
        obs, info = self.observe()
        info['action'] = action
        return obs, reward, self.done, info
            
    def render(self, mode='human', close = False):
        profit = self.net_worth - self.initial_worth
        print('Step: {}'.format(self.current_step))
        print('Net Worth: {}'.format(self.net_worth))
        print('Profit: {}'.format(profit))
        
def create_stock_env(dfs, train=True,balance = 10000,shares=None):
    params = {
        'num_stocks' : len(dfs),
        'balance_normal' : 1000000,
        'shares_normal' : 1000,
        'price_normal': 100,
        'num_prev' : 10,
        'balance': balance,
        'shares_held': shares
    }
    return StockEnv(dfs, params, train)
