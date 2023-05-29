# :red_car: Reinforcement Learning in a 2D Racing Game

In this project we are going to implement a Deep Reinforcement Learning algorithm in a 2D Racing car game.
The goal of the project is to be able to build and train an autonomous AI that is capable of driving along the track. 
We are going to focus on the Q-Learning Algorithm because of its simplicity. 

## Using the code

You can use the jupyter version of the project with the file ```AsynchronousDoubleDeepQLearning.ipynb```.
The code is quite long but is splitted in several classes that I will explain in this file.

Other useful documents can be found in the **Resources** folder.

The **Model** folder contains a trained model that achieved good results in regard to our problem.

The code uses several famous python libraries, one should make those imports:

```
import pygame
import numpy as np
import time
import math
import random
import tensorflow as tf
from tensorflow import keras
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import os
```

## Explication of Reinforcement Learning

Reinforcement Learning is one of the three main classes of problems in Machine Learning: we have supervised learning (classification, regression), unsupervised learning (autoencoder, PCA, ...), and reinforcement learning. Reinforcement learning involves an environment and an agent. The agent performs an action **A**, gets rewarded with reward **R**, and changes to another state **S**. This concept is explained in the following diagram.

![image](https://user-images.githubusercontent.com/66775006/215762340-583f3da4-83fa-4e6a-b76f-f282b253cfb2.png)

## Car Environment

First of all, we need an environment to work with. The environment in Reinforcement Learning contains all information needed for the agent to learn. 
Practically, an environment stores its own state. The change of state happens when calling the `step()` method of the environment which changes the previous state to a next state according to the internal rules of the environment. A reinforcement learning environment canonically returns a tuple of information for the agent. Like the environments made by OpenAI Gym (https://github.com/openai/gym), our environment will return the following information for every call of the `step()` method:

- **observation**: the state of the environment before taking an action 
- **action**: the action taken 
- **reward**: the reward that the agent got in regard of its *(action, state)* pair
- **new_observation**: the new state returned by the environment in consequence of the action
- **done**: a boolean indicating if the new state is terminal or not. 

Additionally an environment should be equipped with a `reset()` method which allows for resetting its state towards the default state. This is very useful when an episode terminates. An episode is a series of steps from the environment until it crosses a terminal state.

For our environment, we used a modified version of a race car game that I found on GitHub, here is the link to this project.

GitHub: https://github.com/techwithtim/Pygame-Car-Racer

Youtube: https://www.youtube.com/watch?v=L3ktUWfAMPg&list=PLzMcBGfZo4-kmY7Nh4kI9kPPnxJ5JMRPj&index=2

Our environment features a Graphical User Interface (GUI) and helps us visualize the result of our training on our agent. The GUI is displayed in the following picture.

![Circuit ](https://user-images.githubusercontent.com/66775006/215759383-94b139d4-76b7-4b7f-a0f6-51e71eac0a16.png)

One can see a car in a racetrack that is subdivided in several colored boxes. Each of the boxes corresponds to a type of racetrack part. For example, red parts correspond to straight parts while purple parts correspond to left turns. 

For simplicity and readability, our environment uses two classes: the `PlayerCar` and the `car_environment`. The compressed code is displayed in the following part:

### PlayerCar class

```
class PlayerCar:

    def __init__(self, max_vel, rotation_vel, track_mask):

    def move(self):
    
    def handle_collision(self):

    def reset(self):
       
    def Inputs(self, cluster_angle):
```

### car_environment class

```
class car_environment():

    def __init__(self):
     
    def checkCluster(self):
       
    def move_player_robot(self, action):
         
    def reward(self):
        
    def reset(self):

    def step(self, action, human):
     
    def draw(self): #(win, images, player_car):
 
    def close(self):
```

## Q-Learning Agent

Now that we have our environment, we can start the code of the agent. The principle of Q-Learning is quite simple: at every step the agent computes the *q-value* of all the possible actions, which corresponds to the value of each choice. Then, the agent takes the best choice if it wants to maximize its immediate reward, or takes another one if it has to explore more possibilities. We are also using a neural network as a function approximator (this is called DQN for Deep Q Network). The diagram of DQN is displayed below:

![Q-Learning](https://user-images.githubusercontent.com/66775006/215767109-4a94e70b-745b-419b-8de6-5f3c0d7f4faf.png)

When updating the weights of our model, we use the Bellman equation as follow:

![Equation](https://user-images.githubusercontent.com/66775006/215766980-c8ea0673-de2d-4a1f-95de-82d8a6d92f5e.png)

Before we display the code of our Agent, I have to underline the tricks that are used in this version of DQN and that are often cited in research papers:
- **Epsilon-greedy strategy**: as explained before, exploration of the environment is a key element and this allows for random exploration with a probability epsilon
- **Double Networks**: two neural networks: a target rarely updated, and another model updated frequently are used in order to improve stability of learning.
- **Experience replay**: experience from past steps is stored in the memory and used for fitting the model with stochastic gradient descent.

Here is the code of our DQN Agent:

```
class minDQN_Agent():
    
    def __init__(self, observation_space, action_space):
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.05
        self.decay = 0.003

        # Main Model (updated every 4 steps)
        self.model = self.createModel(observation_space, action_space)
        # Target Model (updated every 100 steps)
        self.target_model = self.createModel(observation_space, action_space)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=50_000)
        
        self.learning_rate = 0.7 # Learning rate
        self.discount_factor = 0.618

        self.MIN_REPLAY_SIZE = 1000

        self.target_update_counter = 0

        self.steps_to_update_target_model = 0

        self.episode = 0
        self.total_training_rewards = 0
        
        self.RANDOM_SEED = 5
        tf.random.set_seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        
    def setEpsilon(self, start_epsilon, min_epsilon, decay):
        self.epsilon = start_epsilon
        self.max_epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.episode = 0
        self.total_training_rewards = 0
    
    def createModel(self, state_shape, action_shape):
        NNlearning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=(state_shape,), activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(32, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(16, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=NNlearning_rate), metrics=['accuracy'])
        return model
    
    def getAction(self, state):
        random_number = np.random.rand()
        if random_number <= self.epsilon:
            # Explore
            action = np.random.randint(0,env.action_space)
        else:
            # Exploit best known action
            predicted = self.model.predict([state])
            action = np.argmax(predicted)
        return action
        
        
    def train(self, experience):
        
        observation, action, reward, new_observation, done = experience
        self.replay_memory.append([observation, action, reward, new_observation, done])
        self.total_training_rewards += reward
        
        self.steps_to_update_target_model += 1
        
        if done:
            self.episode += 1
            self.total_training_rewards = 0
        
        if self.steps_to_update_target_model >= 100 and done:
            self.target_model.set_weights(self.model.get_weights())
            self.steps_to_update_target_model = 0
                
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.episode)
        
        if self.steps_to_update_target_model % 4 == 0 or done:
            if len(self.replay_memory) < self.MIN_REPLAY_SIZE:
                return

            batch_size = 64 * 2
            mini_batch = random.sample(self.replay_memory, batch_size)
            current_states = np.array([transition[0] for transition in mini_batch])
            current_qs_list = self.model.predict(current_states)
            new_current_states = np.array([transition[3] for transition in mini_batch])
            future_qs_list = self.target_model.predict(new_current_states)

            X = []
            Y = []
            for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
                if not done:
                    max_future_q = reward + self.discount_factor * np.max(future_qs_list[index])
                else:
                    max_future_q = reward

                current_qs = current_qs_list[index]
                current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q

                X.append(observation)
                Y.append(current_qs)
            self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
```

### Neural network

For our neural network we have the following architecture with only 3 hidden layers equipped with *relu* activation functions:

```
Layer (type)                 Output Shape              Param #   
=================================================================
dense_40 (Dense)             (None, 24)                192       
_________________________________________________________________
dense_41 (Dense)             (None, 32)                800       
_________________________________________________________________
dense_42 (Dense)             (None, 16)                528       
_________________________________________________________________
dense_43 (Dense)             (None, 5)                 85        
=================================================================
Total params: 1,605
Trainable params: 1,605
Non-trainable params: 0
```

### Agent actions

We are now going to explain the different actions possible for the agent. As for many race games, the car is fully controllable with the direction keys:

![keys](https://user-images.githubusercontent.com/66775006/215758287-b4b956c2-6d9c-40a9-932b-a46cf5ab7c2e.png)

All actions possible are thus combinations of those actions: for instance pressing both the left key and the forward key makes the car both increase its velocity and turn to the left.

### Input of the Neural Network

For inputs of the neural network, we used a very simple yet effective method called Lidar. This method uses raycasting in several directions and gets the distances between the circuit limits and the car for each corresponding direction. A visualization of this technique is shown in the following pictures, when one can see the car and the different laser beams represented in red color.

![IMe](https://user-images.githubusercontent.com/66775006/215752818-6acd229d-3782-4e98-9da1-8c39b33325df.jpg)

### Rewards

Now that we have the environment, the actions for the agent and the inputs of the model, we need a way to tell the agent if its actions are rewarded in regard to the current state. 

We choose here to be very simple and to assign the reward to be proportional to the distance traveled in the right direction. The angle of the right direction depends on the part of the track the car is in, thanks to the `checkCluster()` method.

The part of the code that deals with the speed reward is written just below:

```
def rewardSpeed(self):
        roadAngle = self.checkCluster()
        radians = math.radians(self.player_car.angle - roadAngle)
        reward = self.player_car.vel*math.cos(radians)/self.player_car.max_vel
        return reward
```

The reward below ranges from *-1* to *+1*. However we also added a negative reward that equals *-10* every time the agent crashes against a wall. This event also triggers the termination of the episode, encouraging the agent to avoid walls. 

We also added a negative reward called reward distance calculated with the code below. Its purpose is to decrease the total reward when the car approaches the walls, but to do nothing when the car is at the center of the road.

```
def rewardDistance(self):
    lidarPoints = self.player_car.lidarMinDistance()
    distances = []
    center = (self.player_car.centerx, self.player_car.centery)
    for i in range(len(lidarPoints)):
        point = lidarPoints[i]
        dist = math.dist(center, point)
        distances.append(dist)

    roadRadius = 36
    minimumDistance = min(distances)

    reward = abs(1 - (minimumDistance/roadRadius))
    return reward
```

## Multi Agent

When running the code above, I realized that while learning was indeed happening, the python program was slow, even with `tensorflow-gpu` activated. I then decided to make some modification to the algorithm in order to further reduce computing time. 

The first thing I did was to create a copy of the environment without GPU thanks to the class `car_environment_blind` so that we save some display time. Then, I implemented an DQN algorithm that used several agent-environments pairs, and computed the `.predict()` methods simultaneously for outputting action values. This change is paramount because tensorflow's neural networks are not optimized for predicting single values.

Those changes helped me greatly reduce learning time.

## Training 

The training is done with the following piece of code. The architecture of the environment and agent classes makes the training very simple. 

```
env = car_environment_blind(-1)

observation_space = env.observation_space
action_space = env.action_space

print("Observation Space : " + str(observation_space))
print("Action Space : " + str(action_space))

multi_agent = MultiDQN_Agent(env)

print("Start of exploration phase\n")

train_episodes = 2000

rewardsData = []
while multi_agent.episode < train_episodes:
    
    experiences = multi_agent.getAction()
    multi_agent.train(experiences)
    
print("\nEnd of exploration phase\n")
```

### Saving and reloading a model

Once the model was trained sufficiently I can now save it in the hard drive along with its memory of past events in order to be able to watch the results later and even continue training.

This part of the code allowed the user to save every useful information of the model in a new folder.

Saving the model:
```
directory = "Wall_avoid_1x^2_2500"
    
# Parent Directory path 
parent_dir = "Models"
    
# Path 
path = os.path.join(parent_dir, directory) 
    
os.mkdir(path) 
print("Directory '% s' created" % directory) 
    
path = parent_dir + "/" + directory + "/"

target_model = multi_agent.target_model
target_model.save(path + "target")

model = multi_agent.model
model.save(path + "model")

memory = list(multi_agent.replay_memory)
with open(path + 'memory.txt', 'w') as f:
    for e in multi_agent.replay_memory:
        f.write(str(e) + '\n')
        
# Saving data:
data = multi_agent.rewardsData
with open(path + 'rewards_data.txt', 'w') as f:
    for e in data:
        f.write(str(e) + '\n')
        
print("Done saving model: " + directory)
```


## Results

After training, we can plot the graph of the evolution of reward over time:

![Raw_Reward_2500](https://github.com/Jean-BaptisteAC/RL-RacingCar/assets/66775006/cad3205b-76ed-4a5d-bf5d-ca66c6d7be32)

Since the agent explores its environment, it sometimes makes wrong decisions and crashes against a wall. Its behavior is not very consistent thus resulting in a very noisy graph. Lets average the previous graph in order to better understand what is going on:

![Comparison_rolling](https://github.com/Jean-BaptisteAC/RL-RacingCar/assets/66775006/0b1073b6-2c71-4409-b139-a8cbac364b5f)

This result is much more understandable, we can witness a very strong and steady progression of the agent along the episodes.
One major conclusion of this graph is that the use of the distance reward significantly improves the performance and reliability of our model.

The following GIF shows one of the best runs of the Multi DQN Agent on the environment GUI: 

![RL Car](https://github.com/Jean-BaptisteAC/RL-RacingCar/assets/66775006/595c8317-ae9e-4370-811a-987bfbfb030a)

One can witness that the agent tends to avoid the walls and maximize its velocity as much as possible, while finishing the track in a very short time by taking sharp corners.

## Conclusion of the project

While results have been quite difficult to obtain due to the many hyperparameters to tune, I am very happy with the final result. The training time is quite small (less than a few hours for the final result shown in the GIF). I managed to train an AI to learn how to drive almost perfectly and with a behavior that is very similar to humans.
