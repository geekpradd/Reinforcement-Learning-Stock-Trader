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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.layers import Dense, Concatenate, Lambda, Activation
from tensorflow.keras import Input
from tensorflow import convert_to_tensor as convert
from collections import deque
import pickle
path_base = 'stable/'

from stable_baselines.common.vec_env import DummyVecEnv,VecCheckNan
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG

from .env import create_stock_env,softmax

class CustomDDPGPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDDPGPolicy, self).__init__(*args, **kwargs, reuse = True,
                                           layers=[128, 64],
                                           layer_norm=True,
                                           feature_extraction="lnmlp")

def DDPGgive_results(files,balance,shares=None):  
    env = create_stock_env(files,train=False,balance =balance,shares = shares)
    max_steps = env.max_steps - env.num_prev
    env = DummyVecEnv([lambda: env])
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(0, 2)
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=1, desired_action_stddev=0.1, adoption_coefficient=1.01)
    model = DDPG(CustomDDPGPolicy, env, verbose=0, param_noise=param_noise, action_noise=action_noise)

    # model = DDPG.load("/home/harshit/Documents/itsp-trade agent/Reinforcement-Learning-Stock-Trader/WebPortal/StockApp/Stock_stable.zip",env=env)
    model.learn(total_timesteps=100)
    profit=0
    profitst = np.zeros((max_steps-1,2))
    actionst = np.zeros(( n_actions//2,max_steps-1,2))
    shares = np.zeros((len(files),max_steps-1,2))
    obs = env.reset()
    for i in range(max_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        actionst[:,i,1] = -info[0]['action'][0][0:n_actions//2] + info[0]['action'][0][n_actions//2:]
        actionst[:,i,0] = i
        shares[:,i,1] = info[0]['shares_held']
        shares[:,i,0] = i
#         print('a',action)
        profit += rewards
        profitst[i] = [i,profit]
        if dones:
            break
    print(info[0]['action'][0])
    print(actionst)
    return profitst.tolist(),shares.tolist(),actionst.tolist()

