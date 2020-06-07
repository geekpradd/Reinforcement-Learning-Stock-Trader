#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from gym import spaces
import numpy as np
import pandas as pd
import json
import datetime as dt

MAX_Money = 10000
num = 1
class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self,dfs, train, number=1, **kwargs):
        super(StockEnv,self).__init__()
        self.train = train
        self.MAX_shares = 2147483647
        self.Min_Brokerage = 30
        self.count = number
        num = number
        self.Brokerage_rate = 0.001
        
        if "balance" in kwargs.keys():
            Max_Money = kwargs["balance"]
        if "Max_Shares" in kwargs.keys():
            self.MAX_shares = kwargs["Shares"]
        if "Broke_limit" in kwargs.keys():
            self.Min_Brokerage = kwargs["Broke_limit"]
        if "Broke_rate" in kwargs.keys():
            self.Brokerage_rate = kwargs["Broke_rate"]
        
        self.dfs = dfs
        self.action_space = spaces.Box(low = np.array([-1]), high = np.array([1]), dtype = np.float16)
        lower = [0]*number
        higher = [1]*number
        self.observation_space = spaces.Box(low=np.array(lower),high=np.array(higher),dtype=np.float32)
    
    def _get_prices(self):
#         print ("Day {0}".format(self.df.loc[self.current_step,"Date"]))
#         print ("low: {0} high: {1}".format(self.df.loc[self.current_step,"Open"],self.df.loc[self.current_step,"Close"]))
        return np.array([np.random.uniform(df.loc[self.current_step,"Open"], df.loc[self.current_step,"Close"]) for df in self.dfs])
    
    def _observe(self, prices):
        frame = prices
        frame = frame / self.highest_price
        info = {
            'balance' : self.balance,
            'highest_price': self.highest_price,
            'current_price': self.current_prices,
            #'time': self.df.loc[self.current_step,'time_stamp'],
            'shares_held': self.shares_held,
            'max_worth': self.max_net_worth,
            'broke_limit': self.Min_Brokerage,
            'broke_rate': self.Brokerage_rate
        }
        
        return frame, info
        
    def reset(self,balance = MAX_Money,initial_shares = np.zeros((num, )),**kwargs):
        if "balance" in kwargs.keys():
            Max_Money = kwargs["balance"]
        if "Max_Shares" in kwargs.keys():
            self.MAX_shares = kwargs["Shares"]
        if "Broke_limit" in kwargs.keys():
            self.Min_Brokerage = kwargs["Broke_limit"]
        if "Broke_rate" in kwargs.keys():
            self.Brokerage_rate = kwargs["Broke_rate"]
        
        if self.train:
            self.current_step = np.random.randint(0,len(self.dfs[0].loc[:,'Open'].values)-1)
        else:
            self.current_step = 0
       
        self.balance = balance
        self.shares_held = initial_shares
        self.current_prices = self._get_prices() 
        self.net_worth = self.balance + sum(initial_shares*self.current_prices)
        self.initial_worth = self.net_worth
        self.max_net_worth = self.net_worth
        self.highest_price = np.max(self.current_prices)
        frame,_ =  self._observe(self.current_prices)
        return frame
    
    def _broke(self,amount):
        return max(amount * self.Brokerage_rate,self.Min_Brokerage)
    
    def _take_action(self, action_vector):
        self.current_prices = self._get_prices()
        self.highest_price = max(self.highest_price,np.max(self.current_prices))
        action_vector  = action_vector*self.MAX_shares
        for i in range(self.count):
            if action_vector[i] < 0:
                # sell
                action_vector[i] = -1*action_vector[i]
                if action_vector[i] > self.shares_held[i]:
                    action_vector[i] = self.shares_held[i]
                amount_gained = action_vector[i]*self.current_prices[i]
                broke = self._broke(amount_gained)
                amount_gained -= broke
                if self.balance + amount_gained < 0:
                    a1 = np.floor(self.balance/((self.Brokerage_rate-1)*self.current_prices[i]))
                    action = np.floor(-(self.balance-self.Min_Brokerage)/self.current_prices[i])
                    if self._broke(a1*self.current_prices[i]) == a1*self.current_prices[i]*self.Brokerage_rate:
                        action_vector[i] = max(a1,action_vector[i])
                    action_vector[i] = max(action_vector[i],0)
                    amount_gained = action_vector[i]*self.current_price
                    amount_gained -= self._broke(amount_gained)
                self.balance +=amount_gained
                self.shares_held[i] = self.shares_held[i]-action_vector[i]
            elif action_vector[i]>0:
                #buy
                amount_required = self.current_prices[i]*action_vector[i] + self._broke(self.current_prices[i]*action_vector[i])
                if amount_required > self.balance:
                    a1 = np.floor(self.balance/((self.Brokerage_rate+1)*self.current_prices[i]))
                    action_vector[i] = np.floor((self.balance-self.Min_Brokerage)/self.current_prices[i])
                    if self._broke(a1*self.current_prices[i]) == a1*self.current_prices[i]*self.Brokerage_rate:
                        action_vector[i] = max(a1,action_vector[i])
                    action_vector[i] = max(action_vector[i],0)
                    amount_required = action_vector[i]*self.current_prices[i]
                    amount_required -= self._broke(amount_required)
                self.balance -= amount_required
                self.shares_held[i] += action_vector[i]
        reward = self.balance + sum(self.shares_held* self.current_prices) - self.net_worth
        self.net_worth = self.balance + sum(self.shares_held* self.current_prices)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        return reward, self.current_prices
            
    def step(self, action):
        reward, prices = self._take_action(action)
        self.current_step+=1
        if self.current_step > len(self.dfs[0].loc[:,'Open'].values)-1:
            self.current_step = 0
        
        done = self.net_worth<=0
        obs, info = self._observe(prices)
        
        return obs, reward, done, info
    
    def render(self, mode='human', close = False):
        profit = self.net_worth - self.initial_worth
        print(f'Step: {self.current_step}')
        print(f'Net Worth:{self.net_worth}')
        print(f'Profit: {profit}')


def create_stock_env(locations, train=True):
    dfs = [pd.read_csv(location) for location in locations]
    for df in dfs:
        print(df.shape)
    return StockEnv(dfs, train, len(locations)), dfs[0].shape[0]


# In[ ]:




