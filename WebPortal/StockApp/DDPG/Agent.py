import gym
from gym import spaces
from matplotlib import pyplot as plt
import time
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import random
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Activation
from tensorflow.keras import Input
from tensorflow import convert_to_tensor as convert
import pickle
from .env import create_stock_env

COLAB = False
if not COLAB:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
path_base = '/content/drive/My Drive/Stock/'
import os

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
        
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class Actor:
    def __init__(self, params):
        self.output_range = params["output_range"]
        self.state_dimensions = params["state_dimensions"]
        self.action_dimensions = params["action_dimensions"]
        self.cap = params['cap']
        self.tau = params['tau']
        self.online_actor = self.build_model()
        self.target_actor = self.build_model()
        
    def build_model(self):
        inputs = Input(shape=(self.state_dimensions, ))
        x = Dense(256, activation = 'relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = Dense(128, activation = 'relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = Dense(64, activation = 'relu')(x)
        s = keras.layers.BatchNormalization()(x)
        sell = Dense(self.action_dimensions, activation = 'sigmoid')(x)
        buy = Dense(self.action_dimensions, activation = 'sigmoid')(x)
        final_buy = Activation(tf.keras.activations.softmax)(buy)*tf.math.minimum(self.cap, tf.reduce_sum(buy, axis = -1, keepdims = True))
        model = keras.Model(inputs = inputs, outputs = tf.concat([sell, final_buy], axis = -1))
        # model.summary()
        return model
    
    def online_get_action(self, state):
    	return self.online_actor(convert(state))

    def target_get_action(self, state):
        return self.target_actor(convert(state))

    def save(self):
        self.online_actor.save(path_base + 'online_actor.h5')
        self.target_actor.save(path_base + 'target_actor.h5')
    
    def load(self):
        self.online_actor = keras.models.load_model(path_base + 'online_actor.h5')
        self.target_actor = keras.models.load_model(path_base + 'target_actor.h5')

    def merge(self):
        self.target_actor.set_weights(self.tau*np.array(self.online_actor.get_weights())
                                                                    + (1-self.tau)*np.array(self.target_actor.get_weights()))

class Critic:
    def __init__(self, params):
        self.state_dimensions = params["state_dimensions"]
        self.action_dimensions = params["action_dimensions"]
        self.optimizer = params["critic_optimizer"]
        self.tau = params['tau']
        self.critic_online = self.build_model()
        self.critic_target = self.build_model()
        self.critic_online.set_weights(self.critic_target.get_weights())

    def build_model(self):
        input_a = Input(shape = (self.state_dimensions, ))
        input_b = Input(shape = (2*self.action_dimensions, ))
        input = Concatenate(axis = -1)([input_a, input_b])
        x = Dense(256, activation = 'relu')(input)
        x = keras.layers.BatchNormalization()(x)
        x = Dense(128, activation = 'relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = Dense(64, activation = 'relu')(x)
        x = keras.layers.BatchNormalization()(x)
        output = Dense(1)(x)
        model = keras.Model(inputs=[input_a, input_b], outputs = output)
        model.compile(loss='mse', optimizer = self.optimizer)
        # model.summary()
        return model

    def save(self):
        self.critic_online.save(path_base + 'critic_online.h5')
        self.critic_target.save(path_base + 'critic_target.h5')

    def load(self):
        self.critic_online = keras.models.load_model(path_base + 'critic_online.h5')
        self.critic_target = keras.models.load_model(path_base + 'critic_target.h5')

    def get_qvalues(self, state_array, action_array, online=True):
        if online:
            return self.critic_online([convert(state_array), convert(action_array)])
        else:
            return self.critic_target([convert(state_array), convert(action_array)])

    def call(self, state_tensor, action_tensor):
        return self.critic_online([state_tensor, action_tensor])
    
    def merge(self):
        self.critic_target.set_weights(self.tau*np.array(self.critic_online.get_weights())
                                                                    + (1-self.tau)*np.array(self.critic_target.get_weights()))

class Agent:
    def __init__(self, params, train = True, resume = True):
        self.train = train
        self.actor = Actor(params)
        self.critic = Critic(params)
        self.buffer = ReplayMemory(params["buffer_size"])
        self.state_dimensions = params["state_dimensions"]
        self.action_dimensions = params["action_dimensions"]
        self.discount = params["discount"]
        self.action_range = params["output_range"]
        self.save_frequency = params["save_frequency"]
        self.batch_size = params["batch_size"]
        self.optimizer = params["actor_optimizer"]
        self.cap = params['cap']
        self.num_steps = 0
        self.noise_func =  OrnsteinUhlenbeckActionNoise(mu=np.zeros(2*params["action_dimensions"]))
        if resume:
            self.load()
        
    def clip_action(self, action):
        action = np.clip(action, 0, self.action_range)
        total = np.sum(action, axis = -1, keepdims = True)
        if not total:
            return action
        action = action*np.minimum(total, self.cap)/total
        return action
    
    def agent_start(self):
            self.prev_state = np.zeros((1,self.state_dimensions))
            self.prev_action = np.zeros((1,2*self.action_dimensions))
                    
    def agent_step(self, reward, observation):    
        self.num_steps+=1
        state = np.reshape(observation, (1, self.state_dimensions))            
        if self.train:
            replay = (self.prev_state, self.prev_action, reward, state)
            self.buffer.append(replay)
            
        action = self.actor.online_get_action(state)
        if self.train:
            action = self.clip_action(action + self.noise_func())
            self.run()
        else:
            action = self.clip_action(action)
        self.prev_action = action
        self.prev_state = state
        return self.prev_action.reshape(2*self.action_dimensions) 
    
    def save(self):
        self.actor.save()
        self.critic.save()
        data = (self.buffer, self.num_steps, self.noise_func)
        with open (path_base + 'auxiliary.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        print('Saved Successfully!')

    def load(self):
        self.actor.load()
        self.critic.load()
        with open (path_base + 'auxiliary.pkl', 'rb') as f:
            data = pickle.load(f)
        self.buffer, self.num_steps, self.noise_func = data
        print('Loaded Successfully!')
    
    def run(self):
        self.num_steps += 1
        size = min(self.batch_size, self.buffer.size)
        batch = self.buffer.sample(size)
        prev_states = np.array([x[0] for x in batch]).reshape((-1, self.state_dimensions))
        prev_actions = np.array([x[1] for x in batch]).reshape((-1, 2*self.action_dimensions))
        rewards = np.array([x[2] for x in batch]).reshape((-1, 1))
        states = np.array([x[3] for x in batch]).reshape((-1, self.state_dimensions))

        target_actions = self.actor.target_get_action(states)
        q_values = self.discount*self.critic.get_qvalues(states, target_actions, False)
        q_values += rewards
        self.critic.critic_online.fit([prev_states, prev_actions], q_values, epochs = 2, verbose=0)

        prev_state_tensor = convert(prev_states)
        prev_action_tensor = convert(prev_actions)
        
        with tf.GradientTape(persistent=True) as tape:
            action = self.actor.online_actor(prev_state_tensor)
            tape.watch(action)
            value = self.critic.call(prev_state_tensor, action)
        gradient = tape.gradient(value, action)
        gradient = -tf.cast(gradient, tf.float32)
        # print(gradient.shape)
        # print(np.max(gradient[0]), 'ankit')
        gradient_actor = tape.gradient(action, self.actor.online_actor.trainable_weights, gradient)
        # print(tape.gradient(action, self.actor.online_actor.trainable_weights)[11].shape, 'ankit')
        # print(len(gradient_actor))
        # print(len(grad))
        # gradient_actor = list(np.array(gradient_actor)/size)
        # print(np.max(gradient_actor[0]))

        self.optimizer.apply_gradients(zip(gradient_actor, self.actor.online_actor.trainable_weights))
        self.critic.merge()
        self.actor.merge()
        del tape

        # if self.num_steps % self.save_frequency == 0:
        #     self.save()




AGENT_PARAMS = {
	"output_range": 1,
	"state_dimensions" : 0,
	"action_dimensions": 2,
	"critic_optimizer": tf.keras.optimizers.Adam(learning_rate = 0.01),
	"actor_optimizer": tf.keras.optimizers.Adam(learning_rate = 0.005),
	"batch_size": 64,
	"buffer_size":100000,
	"discount": 0.99,
	"tau": 0.001,
	"save_frequency": 10000,
	'cap' : 0.9,
}


def DDPGgive_results(files,balance,shares=None):  
	env = create_stock_env(files,train=False,balance =balance,shares = shares)
	max_steps = env.max_steps - env.num_prev
	n_actions = env.action_space.shape[-1]
	n_states = env.observation_space.shape[-1]
	AGENT_PARAMS["state_dimensions"] = n_states
	AGENT_PARAMS["action_dimensions"] = n_actions//2
	model = Agent(params = AGENT_PARAMS, train = True, resume = False)
	for _ in range(1):
		obs = env.reset()
		model.agent_start()
		action = model.agent_step(0,obs)
		for i in range(min(500,max_steps)):
			obs, rewards, dones, info = env.step(action)
			action = model.agent_step(rewards,obs)
			if dones:
				break
	model.train = False

	profit=0
	profitst = np.zeros((max_steps-1,2))
	actionst = np.zeros(( n_actions,max_steps-1,2))
	shares = np.zeros((len(files),max_steps-1,2))
	obs = env.reset()
	model.agent_start()
	action = model.agent_step(0,obs)
	for i in range(max_steps):
		obs, rewards, dones, info = env.step(action)
		actionst[:,i,1] = action
		actionst[:,i,0] = i
		shares[:,i,1] = info['shares_held']
		shares[:,i,0] = i
		#         print('a',action)
		profit += rewards
		profitst[i] = [i,profit]
		if dones:
		    break
		action = model.agent_step(rewards,obs)

	return profitst.tolist(),shares.tolist(),actionst.tolist()