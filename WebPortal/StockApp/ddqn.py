#!/usr/bin/env python
# coding: utf-8

# In[54]:

import gym
from gym import spaces
from matplotlib import pyplot as plt
import time
from tqdm import tqdm_notebook
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from math import floor, ceil
import random
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Activation, LSTM
from tensorflow.keras import Input
from tensorflow import convert_to_tensor as convert
import pickle
COLAB = False
if not COLAB:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import os
path_base = 'StockApp/models/'



# In[80]:


class StockEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, params, train = True):
        super(StockEnv,self).__init__()
        
        self.min_brokerage = params['min_brokerage']
        self.brokerage_rate = params['brokerage_rate']
        self.df = df
        self.state_dimensions = 6
        self.shares_normal = params['shares_normal']
        self.train = train

        self.max_steps = len(self.df.loc[:, "Open"])
        self.action_space = spaces.Box(low = -1, high = 1, shape =  (1, 1), dtype = np.float32)
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (1, self.state_dimensions), dtype = np.float32)

    def reset(self, initial_balance = 10000, shares_held = 100):
        self.start_balance = initial_balance 
        if self.train:
            self.current_step = np.random.randint(0, self.max_steps)
        else:
            self.current_step = 0
        self.balance = initial_balance
        self.shares_held = shares_held
        if self.shares_held is None:
            self.shares_held = 0
        self.current_price = self.get_price()
        self.all_in_shares = initial_balance/self.current_price + shares_held
        self.net_worth = self.balance + (self.shares_held*self.current_price)
        self.initial_worth = self.net_worth
        self.max_net_worth = self.net_worth
        self.done = False
        self.frame = np.zeros((1, self.state_dimensions))
        self.info = {
            'current_step' : self.current_step,
            'current_price': self.current_price,
            'net_worth' : self.net_worth,
            'max_net_worth': self.max_net_worth,
            'shares_held' : self.shares_held,
            'balance' : self.balance,
        }
        return self.observe()
        
    def get_price(self):
        return np.random.uniform(self.df.loc[self.current_step,"Low"], self.df.loc[self.current_step,"High"]) 
      
    def observe(self):
        self.frame[0, 0:4] = np.array([self.df.loc[self.current_step,'Open'],self.df.loc[self.current_step,'High'],self.df.loc[self.current_step,'Low'],self.df.loc[self.current_step,'Close']])/self.balance
        self.frame[0, 4] = self.shares_held/self.shares_normal
        self.frame[0, 5] = self.balance/self.start_balance
        self.info = {
            'current_step' : self.current_step,
            'current_price': self.current_price,
            'net_worth' : self.net_worth,
            'max_net_worth': self.max_net_worth,
            'shares_held' : self.shares_held,
            'balance' : self.balance
        }
        return self.frame, self.info
    
    def update_worth(self, reward):
        self.net_worth += reward
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

    def update_balance(self, action):
        self.balance -= action*self.current_price

    def update_shares(self, action):
        self.shares_held += action

    def take_action(self, action):
        self.current_price = self.get_price()
        max_buyable = self.balance/self.current_price
        max_sellable = self.shares_held
        if action >= 0:
            action *= max_buyable
            action = floor(action)
        else:
            action *= max_sellable
            action = ceil(action)
            
        if self.shares_held == 0 and action < 0:
            reward = 0
            print ("Invallid sell action")
        else:
            self.update_balance(action)
            self.update_shares(action)
            reward = self.balance + (self.shares_held * self.current_price) - self.net_worth
            self.update_worth(reward)
        return reward
            
    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_steps or self.done:
            self.done = True
            return np.zeros((1, self.state_dimensions)), 0, self.done, self.info

        reward = self.take_action(action)
        self.done = self.net_worth <= self.initial_worth*0.05
        if self.done:
            print('snap')
        obs, info = self.observe()
        return obs, reward, self.done, info
    
    def render(self, mode='human', close = False):
        profit = self.net_worth - self.initial_worth
        print('Step: {}'.format(self.current_step))
        print('Net Worth: {}'.format(self.net_worth))
        print('Profit: {}'.format(profit))
        
def create_stock_env(location, train=True):
    df = pd.read_csv(location).sort_values('Date')

    params = {
        'num_stocks' : 1,
        'min_brokerage' : 30.0,
        'shares_normal' : 10000,
        'brokerage_rate' : 0.001,
    }
    return StockEnv(df, params, train)

def create_from_frame(frame, train=False):
    params = {
        'num_stocks' : 1,
        'min_brokerage' : 30.0,
        'shares_normal' : 10000,
        'brokerage_rate' : 0.001,
    }
    return StockEnv(frame, params, train)

# In[81]:


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

from collections import deque
class ReplayDeque:
    def __init__(self, capacity, element_dimensions):
        self.deque = deque()
        self.capacity = capacity
        self.dim = element_dimensions
        for _ in range(self.capacity):
            self.deque.append(np.zeros((element_dimensions, )))
            
    def add(self, obj):
        copied = np.copy(obj)
        self.deque.append(np.squeeze(copied))
        self.deque.popleft()
        
    def get_last(self, duration):
        entries = list(self.deque)[-duration:]
        shape = list(entries[0].shape)
        shape[:0] = [1, len(entries)]
        res = np.concatenate(entries).reshape(shape)
        return res
    def clear(self):
        self.__init__(self.capacity, self.dim)


# In[82]:


class Agent:
    def __init__(self, params, resume=True, train=True):
        self.epsilon = 1
        self.training = train
        self.epsilon_decay = params["decay"]
        self.epsilon_min = params["min_epsilon"]
        self.discount = params["discount"]
        self.merge_frequency = params["merge_frequency"]
        self.save_frequency = params["save_frequency"]
        self.replay_length = params["replay_length"]
        self.num_actions = params["actions"]
        self.state_dimensions = params["state_dimensions"]
        self.batch_size = params["batch_size"]
        self.optimizer = params["optimizer"]
        self.experience_memory = params["memory"]
        self.buffer = ReplayMemory(self.experience_memory)
        self.past_states = ReplayDeque(self.replay_length, self.state_dimensions)
        self.count = 0
        self.game = 0
        self.input_shape = (self.state_dimensions*self.replay_length ,)
        self.q_network = self.build_network()
        self.target_network = self.build_network()
        if resume:
            self.load_weights()
        
    def clear_memory(self):
        self.past_states.clear()
    def merge_networks(self):
        self.target_network.set_weights(self.q_network.get_weights())
    def build_network(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, activation='relu', 
                        input_shape=self.input_shape))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=self.optimizer)
        return model
    
    def agent_start(self, observation):
        self.past_states.add(observation)
        state = np.reshape(self.past_states.get_last(self.replay_length), (1, -1))
        q_values = np.squeeze(self.q_network.predict(state))
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
        if np.random.random() < self.epsilon and self.training:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(q_values)
        self.prev_state = state
        self.prev_action = action 
        return (action-10)/10
    
    def agent_step(self, reward, observation):
        self.past_states.add(observation)
        state = np.reshape(self.past_states.get_last(self.replay_length), (1, -1))
        self.count += 1

        q_values = np.squeeze(self.q_network.predict(state))
        relay = (self.prev_state, self.prev_action,  reward, state, 0)
        self.buffer.append(relay)
        
        if np.random.random() < self.epsilon and self.training:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(q_values)
        self.prev_state = state
        self.prev_action = action 
        if self.training:
            self.train(self.batch_size)
        
        return (action-10)/10
             
    def save_weights(self):
        self.q_network.save_weights(path_base + "main-normal.h5")
        self.target_network.save_weights(path_base + "target-normal.h5")
        data = (self.buffer, self.count, self.epsilon)
        with open (path_base + 'auxiliary-normal.pkl', 'wb') as f:
            pickle.dump(data, f)
               
    def load_weights(self):
        self.q_network.load_weights(path_base + "main-normal.h5")
        self.target_network.load_weights(path_base + "target-normal.h5")
        # with open (path_base + 'auxiliary-normal.pkl', 'rb') as f:
        #     data = pickle.load(f)
        # self.buffer, self.count, self.epsilon = data
        
    def train(self, count):
        size = min(count, self.buffer.size)
        batch = self.buffer.sample(size)
        input_tensor = np.array([state for state, action, reward, future, terminated in batch]).reshape((-1, self.input_shape[0]))
        output_tensor = self.q_network.predict(input_tensor)
        future_input_tensor = np.array([future for state, action, reward, future, terminated in batch]).reshape((-1, self.input_shape[0]))
        future_out = self.target_network.predict(future_input_tensor)
        for count, (state, action, reward, future, terminated) in enumerate(batch):
            target = output_tensor[count]
            updated = reward
            if not terminated:
                target_vals = future_out[count]
                updated += self.discount*(target_vals[np.argmax(target)])
                
            target[action] = updated
            output_tensor[count] = target 
        
        input_tensor = np.array(input_tensor)
        output_tensor = np.array(output_tensor)
        self.q_network.fit(input_tensor, output_tensor, epochs=1, verbose=0)
        if self.count%self.merge_frequency == 0:
            self.merge_networks()
            
        if self.count%self.save_frequency == 0:
            self.save_weights()
            


def train(agent, env, epochs, profits, balances, shares, actions, steps_per_epoch):
    
    for epoch in range(0, epochs):

        cumm_profit = 0
        observation, info = env.reset()
        shares[epoch, 0] = info['shares_held']
        balances[epoch, 0] = info['balance']
        action = agent.agent_start(observation)
        actions[epoch, 0] = action

        for i in (range(steps_per_epoch)):
#             print (agent.past_states.deque)
            observation, reward, done, info = env.step(action)
#             print ("Observation")
#             print (observation)
#             print ("ok")
            shares[epoch, i+1] = info['shares_held']
            balances[epoch, i+1] = info['balance']
            cumm_profit += reward
            profits[epoch, i] = cumm_profit
            if done:
                print("the end")
                break
            action = agent.agent_step(reward, observation)
            actions[epoch, i+1] = action

        agent.clear_memory()
        print('Completed epoch' + str(epoch))


# In[ ]:

def trail(self):
    tf.keras.backend.set_floatx('float32')
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    params = {"state_dimensions":6, "decay":0.995, "batch_size":32, "merge_frequency": 10000, "replay_length":10, "min_epsilon": 0.1, "save_frequency": 5000, "discount": 0.95,  "actions": 20, "optimizer": optimizer, 
              "memory": 70000}
    agent = Agent(params, resume=True)
    files = 'data/PCG.csv'
    env = create_stock_env(files)
    epochs = 50
    steps_per_epoch = 2000
    profits = np.zeros((epochs, steps_per_epoch+1))
    balances = np.zeros((epochs, steps_per_epoch+1))
    shares = np.zeros((epochs, steps_per_epoch+1))
    actions = np.zeros((epochs, steps_per_epoch+1))
    train(agent, env, epochs, profits, balances, shares, actions, steps_per_epoch)


# In[94]:


    for _ in range(20, 30):
        plt.plot(actions[_])

    plt.legend(list(range(10)))
    plt.show()


# In[58]:


def test(balance, env, agent, shares=0):
    max_steps = env.max_steps
    profitst = np.zeros(max_steps+1)
    pricest = np.zeros(max_steps+1)
    balancest = np.zeros(max_steps + 1)
    sharest = np.zeros(max_steps+1)
    actionst = np.zeros(max_steps+1)
    worthst = np.zeros(max_steps+1)

    profit = 0
    profitst[0] = profit
    observation, info = env.reset(initial_balance = balance, shares_held=shares)
    balancest[0] = info['balance']
    print(info['balance'])
    scale = balance/info["current_price"] 
    pricest[0] = scale*info["current_price"]
    sharest[0] = info['shares_held']
    print (info['shares_held'])
    worthst[0] = info['net_worth']
    print(info['shares_held'])
    action = agent.agent_start(observation)
    actionst[0] = action

    for i in (range(max_steps)):
        observation, reward, done, info = env.step(action)
        profit += reward
        profitst[i+1] = profit
        balancest[i+1] = info['balance']
        pricest[i+1] = scale*info["current_price"]
        sharest[i+1] = info['shares_held']
        worthst[i+1] = info['net_worth']
        if done:
            print ("ober")
            break
        action = agent.agent_step(reward, observation)
        actionst[i+1] = action
          


    return profitst, balancest, sharest, actionst, worthst, pricest


# In[77]:

def wrapper(frame, balance, shares):
    env = create_from_frame(frame, train = False)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    params = {"state_dimensions":6, "decay":0.995, "batch_size":32, "merge_frequency": 10000, "replay_length":10, "min_epsilon": 0.1, "save_frequency": 5000, "discount": 0.95,  "actions": 20, "optimizer": optimizer, 
              "memory": 70000}
    agent = Agent(params, resume = True, train = False)
    return test(balance, env, agent, shares)




class WebInterface:
    def __init__(self, location, train=False, percentage=80, steps=2000, epochs=10):
        #percentage refers to percentage of dataframe that will be used for training
        split = percentage/100
        self.steps = steps
        self.epochs = epochs
        self.train = train
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        params = {"state_dimensions":6, "decay":0.995, "batch_size":32, "merge_frequency": 10000, "replay_length":10, "min_epsilon": 0.1, "save_frequency": 5000, "discount": 0.95,  "actions": 20, "optimizer": optimizer, 
                  "memory": 70000}
        self.agent = Agent(params, resume=True)
        if train:
            self.train_env = create_stock_env(location, train, split)
            self.test_env = create_stock_env(location, False, split)
        else:
            self.test_env = create_stock_env(location, False, 1)
        
    def train(self):
        if not self.train:
            raise Exception("Training is not supported on non training interfaces")
        self.agent.training = True
        
        profits = np.zeros((self.epochs, self.steps+1))
        balances = np.zeros((self.epochs, self.steps+1))
        shares = np.zeros((self.epochs, self.steps+1))
        actions = np.zeros((self.epochs, self.steps+1))
        train(self.agent, self.train_env, self.epochs, profits, balances, shares, actions, self.steps)
        
        return profits, balances, shares, actions
    
    def test(self, balances):
        self.agent.training = False
        return test(balances, self.test_env, self.agent)


# In[ ]:




