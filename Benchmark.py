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

#%% ONE AGENT WITHOUT DISTANCE REWARD
#%%

env = car_environment_blind(0)

observation_space = env.observation_space
action_space = env.action_space

print("Observation Space : " + str(observation_space))
print("Action Space : " + str(action_space))


agent = minDQN_Agent(observation_space, action_space)
agent.model.summary()

rewardsData = []


#%% Training phase

print("Start of exploration phase\n")
start = time.time()

human_control = False
episode = 0
train_episodes = 2000
step = 0

while episode < train_episodes and (time.time() - start < 3600):
    state = env.reset()
    done = False
    while not done:
    
        action = agent.getAction(state)
        next_state, reward, done = env.step(action)
        experience = state, action, reward, next_state, done
        step += 1
        
        if done:
            print('Episode = {}, total reward: {}, , epsilon = {}'.format(episode, round(agent.total_training_rewards, 1), round(agent.epsilon, 3)))
            print("Step:", step)
            rewardsData.append(agent.total_training_rewards)

        agent.train(experience)
        state = next_state
                    
    episode += 1
    
print("\nEnd of exploration phase\n")
finish = time.time()

#%% Displaying Progress

MAX_REWARD = 841.8 # Full Track

data = rewardsData
print("Max reward:", max(data))

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
    ax.margins(x=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)


tsplot(ax, data)

ax.set_title("Reward over time (95% ci)")

plt.show()


weight = 30
if len(data) > weight:
    numbers_series = pd.Series(data)
    windows = numbers_series.rolling(weight)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    
print("Max moving average:", max(moving_averages_list[30:]))
print("Cummulative reward:", np.cumsum(data)[-1])
#%% Saving the model

directory = "One_Agent_60mins"
    
# Parent Directory path 
parent_dir = "Models/Benchmark"
    
# Path 
path = os.path.join(parent_dir, directory) 
    
os.mkdir(path) 
print("Directory '% s' created" % directory) 
    
path = parent_dir + "/" + directory + "/"

target_model = agent.target_model
target_model.save(path + "target")

model = agent.model
model.save(path + "model")

with open(path + 'memory.txt', 'w') as f:
    for e in agent.replay_memory:
        f.write(str(e) + '\n')
        
# Saving data:
data = rewardsData
with open(path + 'rewards_data.txt', 'w') as f:
    for e in data:
        f.write(str(e) + '\n')
        
print("Done saving model: " + directory)

#%% MULTI AGENT WITHOUT DISTANCE REWARD
#%%   
env = car_environment_blind(-1)

observation_space = env.observation_space
action_space = env.action_space

print("Observation Space : " + str(observation_space))
print("Action Space : " + str(action_space))

multi_agent = MultiDQN_Agent(env)
multi_agent.model.summary()

#%%

start_time = time.time() 
print("Start of exploration phase\n")

train_episodes = 1000
step = 0

display = True
while multi_agent.episode < train_episodes and (time.time() - start_time < 3600):
    
    experiences = multi_agent.getAction()
    multi_agent.train(experiences)
    step += multi_agent.n_env
    
    if multi_agent.episode%10==0 :
        if display:
            multi_agent.display_progress()
            print("Step:", step)
            display = False 
    else:
        display = True

print("\nEnd of exploration phase\n")
finish_time = time.time()

#%% Displaying Progress

MAX_REWARD = 841.8 # Full Track

data = multi_agent.rewardsData
print("Max reward:", max(data))

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
    ax.margins(x=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True)


tsplot(ax, data)

ax.set_title("Reward over time (95% ci)")

plt.show()


weight = 30
if len(data) > weight:
    numbers_series = pd.Series(data)
    windows = numbers_series.rolling(weight)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    
print("Max moving average:", max(moving_averages_list[30:]))
print("Cummulative reward:", np.cumsum(data)[-1])

#%% Saving the model

directory = "Multi_Agent_60mins"
    
# Parent Directory path 
parent_dir = "Models/Benchmark"
    
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

#%% MULTI AGENT WITHOUT DISTANCE REWARD 3000 Episodes
#%%   

from Agent import extract_memory_txt
from Agent import extract_reward_data

# Parent Directory path 
parent_dir = "Models"

directory = "Model_3000"

# Path 
path = parent_dir + "/" + directory + "/"

# LoadedModel = keras.models.load_model(path + 'model')
# LoadedTarget = keras.models.load_model(path + 'target')
# LoadedMemory = extract_memory_txt(path)
LoadedRewards_without_distance = extract_reward_data(path)


#%% MULTI AGENT WITH DISTANCE REWARD 3000 Episodes
#%%   

from Agent import extract_memory_txt
from Agent import extract_reward_data

# Parent Directory path 
parent_dir = "Models"

directory = "Wall_avoid_1x^2_2500"

# Path 
path = parent_dir + "/" + directory + "/"

# LoadedModel = keras.models.load_model(path + 'model')
# LoadedTarget = keras.models.load_model(path + 'target')
# LoadedMemory = extract_memory_txt(path)
LoadedRewards_with_distance = extract_reward_data(path)

#%% Displaying Progress



def scale_without_distance(data):
    random_score = 16.56
    player_score = 846.676946
    for i in range(len(data)):
        data[i] = 100*(data[i] - random_score)/(player_score - random_score)
    return data

LoadedRewards_without_distance = scale_without_distance(LoadedRewards_without_distance)

def scale_with_distance(data):
    random_score = 10.925
    player_score = 761.887337
    for i in range(len(data)):
        data[i] = 100*(data[i] - random_score)/(player_score - random_score)
    return data

LoadedRewards_with_distance = scale_with_distance(LoadedRewards_with_distance)

MAX_REWARD = 100

data = scale_with_distance(LoadedRewards_with_distance)
plt.figure(figsize=(12, 5), dpi=80)
plt.plot(data, label="Reward")
plt.plot([MAX_REWARD]*len(LoadedRewards_with_distance), 'r', label="Maximum reward (100%)")
plt.xlabel("Episode")
plt.ylabel("Reward (%)")
plt.title("Reward over time")
plt.xlim([0, len(LoadedRewards_with_distance)])
plt.legend(loc='upper left')
plt.show()



def display_progress(data, title):
    weight = 30
    if len(data) > weight:
        numbers_series = pd.Series(data)
        windows = numbers_series.rolling(weight)
        moving_averages = windows.mean()
        moving_averages_list = moving_averages.tolist()


        plt.plot(moving_averages_list, label=title)
        plt.xlabel("Episode")
        plt.ylabel("Reward (%)")
        plt.title("Rolling-average reward over time")

plt.figure(figsize=(12, 5), dpi=80)
display_progress(LoadedRewards_with_distance, "With distance reward")
display_progress(LoadedRewards_without_distance, "No distance reward")
N = len(LoadedRewards_without_distance)
x = np.linspace(0,N)
plt.plot(x, [MAX_REWARD]*len(x), 'r', label="Human Level")

plt.legend(loc='upper left')
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(12,5))

def tsplot(ax, y, title,**kw):
    
    channels = 20
    
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
    ax.fill_between(x,cis[0],cis[1],alpha=0.3, **kw)
    ax.plot(x,est, label=title,**kw)
    ax.margins(x=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (%)")
    ax.grid(True)

tsplot(ax, LoadedRewards_with_distance, "With distance reward")
tsplot(ax, LoadedRewards_without_distance, "No distance reward")

N = len(LoadedRewards_without_distance)
x = np.linspace(0,N)
ax.plot(x, [MAX_REWARD]*len(x), 'r', label="Human level")
ax.legend()
ax.legend(loc='upper left')
ax.set_title("Reward over time (95% ci)")

plt.show()

#%% MULTI AGENT WITH DISTANCE REWARD 3000 Episodes
#%%   

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
LoadedAgent.setEpsilon(0.05,0.05,0.003)

#%%

print("Start of watching phase\n")
env = car_environment()

human_control = False
episode = 0
watch_episodes = 100

total_reward = 0
best_actions = []
best_traj = []
agent_times = []
minimum = 1000

while episode < watch_episodes and env.run:
    step = 0
    state = env.reset()
    done = False
    actions = []
    traj = []
    while not done and env.run:
    
        action = LoadedAgent.getAction(state)

        next_state, reward, done = env.step(action, human_control)
        total_reward += reward
        experience = state, action, reward, next_state, done
        actions.append(action)
        traj.append([env.player_car.centerx, env.player_car.centery])
        
        step += 1
        
        if done:
            if reward == 100:
                agent_times.append(step/30)
                print(step/30)
                if step/30 < minimum:
                    minimum = step/30
                    best_actions = actions
                    best_traj = traj

            total_reward = 0
    
        state = next_state
         
        if env.run is False:
            break
    episode += 1
    
print("\nEnd of watching phase\n")
env.close()

#%% Save Data

directory = "Wall_avoid_1x^2_2200"
    
# Parent Directory path 
parent_dir = "AI_Data"
    
# Path 
path = os.path.join(parent_dir, directory) 

os.mkdir(path) 
print("Directory '% s' created" % directory) 
    
path = parent_dir + "/" + directory + "/"

# Saving time data:
with open(path + 'times.txt', 'w') as f:
    for e in agent_times:
        f.write(str(e) + '\n')
        
print("Done saving trajectory: " + directory)

# Saving traj:
with open(path + 'Trajectory.txt', 'w') as f:
    for e in best_traj:
        f.write(str(e) + '\n')
        
# Saving actions:
with open(path + 'actions.txt', 'w') as f:
    for e in best_actions:
        f.write(str(e) + '\n')
        
print("Done saving trajectory: " + directory)




#%% Display easyJet and 1x2 2200 trajectories
#%%
from CarEnvironment import car_environment_trajectory
import os

def extract_trajectory_txt(path):
    with open(path + 'Trajectory.txt') as f:
        lines = f.readlines()

    trajectory_data = []
    for i in range(len(lines)):

        line = lines[i][1:-2]
        state = []
        start = 0
        for j in range(len(line)):
            if line[j] == ',':
                state.append(float(line[start:j]))
                break

        state.append(float(line[j+2:-1]))

        trajectory_data.append(state)
        
    return trajectory_data

directory = "easyJet"
    
# Parent Directory path 
parent_dir = "Players_Data"
    
# Path 
path = parent_dir + "/" + directory + "/"

Ryan_trajectory = extract_trajectory_txt(path)

directory = "Wall_avoid_1x^2_2200"
    
# Parent Directory path 
parent_dir = "AI_Data"   

path = parent_dir + "/" + directory + "/"

AI_trajectory = extract_trajectory_txt(path)

Visualisation = car_environment_trajectory(Ryan_trajectory, AI_trajectory)
while Visualisation.run:
    Visualisation.draw()
