import gym
import numpy as np
import pandas as pd

df = pd.read_csv('./data/AAPL.csv')
env = gym.make('CustomEnvs:stock-v0', df = df)
print(env.action_space)

for i_episode in range(20):
	observation = env.reset()
	for t in range(100):
		env.render()
		# print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
env.close()