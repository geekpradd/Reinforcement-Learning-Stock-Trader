## Reinforcement Learning Based Stock Trader

This project is done under the institute technical summer project of IIT Bombay. The main idea is to investigate different reinforcement learning architectures for stock trading (deep Q learning, policy gradient iteration, deep deterministic policy gradient, LSTM based predictors etc). We have covered the Reinforcement learning specialisation from Coursera. OpenAI Gym will be used for the environment.

The goal as of now is to implement various control problems in OpenAI Gym starting from classic pendulum control to Atari games using ConvNets and then we will implement several models for stock trading proposed in papers and experiment with our own models as well. Stock data will be fetched from Kaggle.

So far DDPG for continous control has been implemented for pendulum control and tested on the stock environment on a single stock which is prone to overfitting. DDQN has been implemented for CartPole and Atari-RAM and DDRQN has been implemented for use in the stock environment.
