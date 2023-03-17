import pygame
import numpy as np
import time
import math
import tensorflow as tf
from tensorflow import keras
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

from CarEnvironment import car_environment, car_environment_blind
from Agent import minDQN_Agent, MultiDQN_Agent

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
        
#%% Training

env = car_environment_blind(-1)

observation_space = env.observation_space
action_space = env.action_space

print("Observation Space : " + str(observation_space))
print("Action Space : " + str(action_space))

multi_agent = LoadedAgent
multi_agent.model.summary()

#17h48-18h48: 1000 episodes
# 3h20 min: 1000ep -> 1900ep
# 22h35 : 00h05 (1h30)
# TOTAL: 5h50
# Epsilon final = 0.052
# + 1h

start_time = time.time()
print("Start of exploration phase\n")
multi_agent.setEpsilon(0.02,0.02,0.003)
train_episodes = 3000

display = True
while multi_agent.episode < train_episodes:
    
    experiences = multi_agent.getAction()
    multi_agent.train(experiences)
    
    if multi_agent.episode%10==0 :
        if display:
            multi_agent.display_progress()
            display = False 
    else:
        display = True

print("\nEnd of exploration phase\n")
finish_time = time.time()


#%% Displaying Progress

MAX_REWARD = 800 # Full Track

data = multi_agent.rewardsData

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
        plt.plot([MAX_REWARD]*len(moving_averages_list), 'r', label='Maximum reward')
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
    ax.plot(x, [MAX_REWARD]*len(x), 'r', label="Maximum reward ")
    ax.margins(x=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)


tsplot(ax, data)

ax.set_title("Reward over time (95% ci)")

plt.show()

#%% Saving the model

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

#%% Loading a model
from Agent import extract_memory_txt
from Agent import extract_reward_data

# Parent Directory path 
parent_dir = "Models"

directory = "Wall_avoid_1x^2_2200"

# Path 
path = parent_dir + "/" + directory + "/"

LoadedModel = keras.models.load_model(path + 'model')
LoadedTarget = keras.models.load_model(path + 'target')
LoadedMemory = extract_memory_txt(path)
LoadedRewards = extract_reward_data(path)

print("Succesfully loaded model:", directory)

env = car_environment()
env.close()
observation_space = env.observation_space
action_space = env.action_space

print("Observation Space : " + str(observation_space))
print("Action Space : " + str(action_space))

LoadedAgent = minDQN_Agent(observation_space, action_space)
LoadedAgent.model = LoadedModel
LoadedAgent.target_model = LoadedTarget
LoadedAgent.replay_memory = deque(LoadedMemory)
LoadedAgent.rewardsData = LoadedRewards
LoadedAgent.setEpsilon(0.02,0.02,0.003)

#%% Showing Stats

data = LoadedRewards

plt.figure(figsize=(12, 5), dpi=120)
plt.plot(data,label ='Reward')

plt.plot([MAX_REWARD] * len(LoadedRewards), 'r', label='Maximum reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.legend()
plt.show()

display_progress(data)

fig, ax = plt.subplots(figsize=(12,5))

tsplot(ax, data)

ax.set_title("Reward over time (95% ci)")

plt.show()

#%% Watch results

print("Start of watching phase\n")
env = car_environment()

human_control = False
episode = 0
watch_episodes = 100

total_reward = 0
episode_actions = []

while episode < watch_episodes and env.run:
    state = env.reset()
    done = False
    actions = []
    while not done and env.run:
    
        action = LoadedAgent.getAction(state)

        next_state, reward, done = env.step(action, human_control)
        total_reward += reward
        experience = state, action, reward, next_state, done
        actions.append(action)
        
        if done:
            print('Episode = {}, total reward: {}, , epsilon = {}'.format(episode, round(total_reward, 1), round(LoadedAgent.epsilon, 3)))
            total_reward = 0
            episode_actions.append(actions)
            actions = []
            
        # LoadedAgent.train(experience)
        state = next_state
         
        if env.run is False:
            break
    episode += 1
    
print("\nEnd of watching phase\n")
env.close()

# best run:
# [0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 2, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 1, 1, 3, 1, 1, 2, 1, 2, 1, 2, 1, 3, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 0, 1, 3, 1, 3, 0, 3, 1, 4, 1, 2, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 1, 3, 3, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 2, 0, 2, 3, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 3, 1, 3, 1, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 3, 1, 1, 1, 4, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 0, 0, 0, 3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 4, 1, 0, 4, 1, 3, 4, 4, 4, 4, 1, 4, 2, 4, 2, 2, 4, 2, 4, 2, 3, 2, 3, 0, 3, 3, 3, 3, 3, 4, 2, 4, 2, 4, 3, 2, 3, 2, 3, 2, 3, 2, 4, 2, 3, 3, 2, 3, 3, 2, 3, 2, 3, 0, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 1, 3, 1, 1, 1, 1, 2, 1, 0, 2, 0, 2, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 2, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 2, 3, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 1, 3, 1, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 2, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 3, 0, 3, 2, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0, 3, 0, 0, 3, 0, 0, 3, 2, 0, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 4, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 3, 1, 4, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1, 1, 1, 3, 1, 1, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2]


