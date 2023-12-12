# RLBGU
## Solutions for homework in BGU RL Curse 372.2.5910  given by Gilad kats
- In assignment 1 we will get to know and experiment with OpenAI Gymnasium, a testbed
for reinforcement learning algorithms containing environments in different difficulty
levels. We will first implement a tabular Q-learning model on a simple environment in Qustion1 : "FrozenLake-v0".
Then in Qustions 2+3, We will move on to a larger scale environment, "CartPole-v1". In Question 2 we will use a NN function
approximator of the Q-value, using the basic DQN algorithm. In Question 3 will try to
improve DQN with one of the state-of the-art algorithms from recent years-DDQN.
NN's are computed with tensorflow

- In assignment 2 we will optimize the policy of the agent directly
with policy gradient methods. We will first implement the basic REINFORCE
algorithm and then transform it to REINFORCE with baseline and an Actor-Critic algorithm.
we again solve gymnasium's "CartPole-v1" environment

- In assignment 3 we take the REINFORCE algorithm we implemented in hw2 and apply it to solve "CartPole-v1" ,"MountainCarContinuoues-v0"
  and "Acrobot-v1". Then, after the models are trained, We try and see how Fine tuning can help improve preformeace.
  We take the trained model for one env, freeze the output layer and than train it on another env and check how different it is than the original training. 
