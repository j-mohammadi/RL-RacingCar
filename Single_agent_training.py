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

from CarEnvironment import car_environment, car_environment_blind
from Agent import Simple_DQN_Agent ,minDQN_Agent, MultiDQN_Agent

#%% GPU

print("GPUs available : " , len(tf.config.list_physical_devices('GPU')))
print("Tensorflow Version : " + str(tf.__version__))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
#%% Creation of env

env = car_environment()
env.close()

observation_space = env.observation_space
action_space = env.action_space

print("Observation Space : " + str(observation_space))
print("Action Space : " + str(action_space))


agent = Simple_DQN_Agent(observation_space, action_space)
agent.model.summary()

rewardsData = []

#%% Training phase

print("Start of exploration phase\n")
env = car_environment()

human_control = False
episode = 0
train_episodes = 1000

while episode < train_episodes and env.run:
    state = env.reset()
    done = False
    while not done and env.run:
    
        action = agent.getAction(state)
        next_state, reward, done = env.step(action, human_control)
        experience = state, action, reward, next_state, done
        
        if done:
            print('Episode = {}, total reward: {}, , epsilon = {}'.format(episode, round(agent.total_training_rewards, 1), round(agent.epsilon, 3)))
            rewardsData.append(agent.total_training_rewards)

        agent.train(experience)
        state = next_state
                    
        if env.run is False:
            break
    episode += 1
    
print("\nEnd of exploration phase\n")
env.close()


#%% Displaying Progress

MAX_REWARD = 841.8 # Full Track

data = rewardsData

plt.figure(figsize=(12, 5), dpi=80)
plt.plot(data)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.show()

def display_progress(data):
    weight = 30
    if len(data) > weight:
        numbers_series = pd.Series(data)
        windows = numbers_series.rolling(weight)
        moving_averages = windows.mean()
        moving_averages_list = moving_averages.tolist()

        plt.figure(figsize=(12, 5), dpi=80)
        plt.plot(moving_averages_list, label="Average reward")
        # plt.plot([MAX_REWARD]*len(moving_averages_list), 'r', label='Maximum reward')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Rolling-average reward over time")
        plt.legend()
        plt.grid(True)
        plt.show()
        
display_progress(data)

fig, ax = plt.subplots(figsize=(12,4))

def tsplot(ax, y,**kw):
    
    channels = 10
    
    data = []
    N = (len(y)//channels)*channels
    y = y[:N]
    for i in range(channels):
        data.append([y[channels*n + i] for n in range(N//channels)])

    data = np.array(data)
    
    x = np.linspace(0,N,N//channels)
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est, label="Average reward",**kw)
    # ax.plot(x, [MAX_REWARD]*len(x), 'r', label="Maximum reward ")
    ax.margins(x=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)


tsplot(ax, data)

ax.set_title("Reward over time (95% ci)")

plt.show()

#%% Saving the model

directory = "Simple_DQN_4h"
    
# Parent Directory path 
parent_dir = "Models"
    
# Path 
path = os.path.join(parent_dir, directory) 
    
os.mkdir(path) 
print("Directory '% s' created" % directory) 
    
path = parent_dir + "/" + directory + "/"

model = agent.model
model.save(path + "model")
        
# Saving data:
data = rewardsData
with open(path + 'rewards_data.txt', 'w') as f:
    for e in data:
        f.write(str(e) + '\n')
        
print("Done saving model: " + directory)