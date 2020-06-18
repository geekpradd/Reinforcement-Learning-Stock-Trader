# Reinforcement Learning Based Stock Trader

This project is done under the Institute Technical Summer project of IIT Bombay. The idea is to investigate various algorithms of Reinforcement Learning to the domain of Stock Trading. 

As part of learning, we have covered Coursera Reinforcement Learning Specialisation offered by University of ALberta. We have read around 10-15 research papers covering a lot of major and minor aspects of Deep Learning and Reinforcement Learning. We explored OpenAI Gym to gain insights into how to setup our stock trading environment. 

The various algorithms we discuss are namely:
* Deep Q Learning
* Deep Double Q Learning
* Duelling Deep Double Q Learning
* Deep Deterministic Policy Gradient
* Deep Recurrent Double Reinforcement Learning

Stock Trading being a really challenging RL Problem, instead of jumping straightaway to the Stock Trading part, we go on a step by step basis starting from lower state space control problems to high state space Atari Games. And then finally to Stock Trading.

## Control Problems

We started with various control problems (lower state space) to test the architectures/algorithms and get a firm grip on the algorithms and Reinforcement Learning as a whole. The problems include - 
* Pendulum
* Cart Pole
* Mountain Car

## Atari Games

Having tried our algos on lower state space problems, we jumped into the Atari Games that have a really large state space.
The games include - 
* Breakout
* Pong

The results are reported below - 



We implemented both Feedforward Neural Nets and Conv Nets for RAM and image versions of the game respectively. But due to lack of computational resources, we were forced to train on the RAM version. Using OpenCV, we obtained the pixels of the game play and made a video out of it which is shown below.

Our best score on Breakout was close to that reported by DeepMind by certainly less than the best score reported by the OpenAI team which is because we couldn't train for long due to lack of computational resources and time.

## Stock Trading

We now shift to the main part of the project i.e. Stock Trading.

We started reading a lot of research papers and articles on algorithms and their applications. We also came across some articles and reports on Stock Trading using DRQN, DDPG and DDRPG algorithms. We explored various aspects of the algorithms and concluded the relevant algorithms.

We began by implementing the environment for Stock Trading using OpenAI Gym. This was a basic single stock version of the environment for ou first DDPG Single Stock Agent. We observed some really good performance that made us rethink the implementaions of the agent and environment. 

Having fixed the errors for single stock environment and the agent, the agent was trained on it and we achieved a decent performance which is reported here.

Having gained some intuition of how things are going on, we modularised out environment code to tackle and multi stock scenario. We developed our DDPG agent with a better architecture and also modularised it to pertain to any multi stock environment. The model was trained on the environment and the results are reported here.

We planned to use Recurent Layers in our model and for that we completed the Sequence Modelling Course offered by DeepLearning.ai on Coursera. We trained a DRQN model on the single stock environment as applying it to a multi stock environment made it really complex to code which was not possible to achieve in these limited number of days. The results are reported here.

The basic idea of using the recurrent layers was to somehow make the agent remember past data so that it can use that information to make informed decisions. This is what LSTM promises to do.
