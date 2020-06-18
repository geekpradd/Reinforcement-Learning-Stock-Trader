import gym
from gym import spaces
import numpy as np
import pandas as pd
import json
import datetime as dt

MAX_Money = 1000
class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self,df, train, volume=True, **kwargs):
        super(StockEnv,self).__init__()
        self.volume = volume
        self.train = train
        self.MAX_shares = 2147483647
        self.Min_Brokerage = 30
        self.Brokerage_rate = 0.001
        
        if "Balance" in kwargs.keys():
            Max_Money = kwargs["Balance"]
        if "Max_Shares" in kwargs.keys():
            self.MAX_shares = kwargs["Shares"]
        if "Broke_limit" in kwargs.keys():
            self.Min_Brokerage = kwargs["Broke_limit"]
        if "Broke_rate" in kwargs.keys():
            self.Brokerage_rate = kwargs["Broke_rate"]
        
        self.df = df
        self.action_space = spaces.Box(low = np.array([-1]), high = np.array([1]), dtype = np.float16)
        self.observation_space = spaces.Box(low=np.array([0,0,0,0]),high=np.array([1,1,1,1]),dtype=np.float16)
    
    def _get_price(self):
#         print ("Day {0}".format(self.df.loc[self.current_step,"Date"]))
#         print ("low: {0} high: {1}".format(self.df.loc[self.current_step,"Open"],self.df.loc[self.current_step,"Close"]))
        return np.random.uniform(self.df.loc[self.current_step,"Open"],self.df.loc[self.current_step,"Close"])
    
    def _observe(self):
        frame = np.array([self.df.loc[self.current_step,'Open'],self.df.loc[self.current_step,'High'],self.df.loc[self.current_step,'Low'],self.df.loc[self.current_step,'Close']])
        frame = frame / self.highest_price
        info = {
            'balance' : self.balance,
            'highest_price': self.highest_price,
            'current_price': self.current_price,
            #'time': self.df.loc[self.current_step,'time_stamp'],
            'shares_held': self.shares_held,
            'max_worth': self.max_net_worth,
            'broke_limit': self.Min_Brokerage,
            'broke_rate': self.Brokerage_rate
        }
        
        return frame, info
        
    def reset(self,balance = MAX_Money,initial_shares = 0,**kwargs):
        if "Balance" in kwargs.keys():
            Max_Money = kwargs["Balance"]
        if "Max_Shares" in kwargs.keys():
            self.MAX_shares = kwargs["Shares"]
        if "Broke_limit" in kwargs.keys():
            self.Min_Brokerage = kwargs["Broke_limit"]
        if "Broke_rate" in kwargs.keys():
            self.Brokerage_rate = kwargs["Broke_rate"]
        
        if self.train:
            self.current_step = np.random.randint(0,len(self.df.loc[:,'Open'].values)-1)
        else:
            self.current_step = 0
        print (self.train)
        print (self.current_step)
        self.balance = balance
        self.shares_held = initial_shares
        self.current_price = self._get_price() 
        self.net_worth = self.balance + initial_shares*self.current_price
        self.initial_worth = self.net_worth
        self.max_net_worth = self.net_worth
        self.highest_price = self.current_price
        frame,_ =  self._observe()
        return frame
    
    def _broke(self,amount):
        return max(amount * self.Brokerage_rate,self.Min_Brokerage)
    
    def _take_action(self,action):
        self.current_price = self._get_price()
        self.highest_price = max(self.highest_price,self.current_price)
        action  = action*self.MAX_shares
        if action < 0:
            # sell
            action = -1*action
            if action > self.shares_held:
                action = self.shares_held
            amount_gained = action*self.current_price
            broke = self._broke(amount_gained)
            amount_gained -= broke
            if self.balance + amount_gained < 0:
                a1 = np.floor(self.balance/((self.Brokerage_rate-1)*self.current_price))
                action = np.floor(-(self.balance-self.Min_Brokerage)/self.current_price)
                if self._broke(a1*self.current_price) == a1*self.current_price*self.Brokerage_rate:
                    action = max(a1,action)
                action = max(action,0)
                amount_gained = action*self.current_price
                amount_gained -= self._broke(amount_gained)
            self.balance +=amount_gained
            self.shares_held = self.shares_held-action
        elif action>0:
            #buy
            amount_required = self.current_price*action + self._broke(self.current_price*action)
            if amount_required > self.balance:
                a1 = np.floor(self.balance/((self.Brokerage_rate+1)*self.current_price))
                action = np.floor((self.balance-self.Min_Brokerage)/self.current_price)
                if self._broke(a1*self.current_price) == a1*self.current_price*self.Brokerage_rate:
                    action = max(a1,action)
                action = max(action,0)
                amount_required = action*self.current_price
                amount_required -= self._broke(amount_required)
            self.balance -= amount_required
            self.shares_held += action
        reward = self.balance + self.shares_held * self.current_price - self.net_worth
        self.net_worth = self.balance + self.shares_held * self.current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        return reward
            
    def step(self,action):
        reward = self._take_action(action)
        self.current_step+=1
        if self.current_step > len(self.df.loc[:,'Open'].values)-1:
            self.current_step = 0
        
        done = self.net_worth<=0
        obs, info = self._observe()
        
        return obs, reward, done, info
    
    def render(self, mode='human', close = False):
        profit = self.net_worth - self.initial_worth
        print(f'Step: {self.current_step}')
        print(f'Net Worth:{self.net_worth}')
        print(f'Profit: {profit}')


def create_stock_env(location, train=True):
    df = pd.read_csv(location)
    df = df.sort_values('Date')
    return StockEnv(df, train), df.shape[0]