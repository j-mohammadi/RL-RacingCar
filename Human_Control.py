from CarEnvironment import car_environment

env = car_environment()
state = env.reset()

actionList = []
actionList += [2]*33

actionList += [1]*45

actionList += [2]*68

actionList += [1]*11

actionList += [2]*80

actionList += [1]*34

actionList += [2]*27

actionList += [3]*23

actionList += [2]*17

actionList += [3]*20

actionList += [2]*33

actionList += [1]*23

actionList += [2]*2

actionList += [1]*20

actionList += [2]*58

actionList += [1]*22

actionList += [2]*48

actionList += [3]*45

actionList += [2]*45

actionList += [1]*24

actionList += [2]*20

actionList += [1]*22

actionList += [2]*75

actionList += [1]*22

actionList += [2]*50

actionList += [3]*46

actionList += [2]*30


human_control = False
i = 0
total_reward = 0
human_trajectory = []
while env.run and i < len(actionList):
    next_state, reward, done = env.step(actionList[i], human_control)
    total_reward += reward
    x, y = env.player_car.centerx, env.player_car.centery
    human_trajectory.append([x,y])
    if done:
        print("Total number of frames :" , i)
        print("Total convertedtime :" , i/60, 's')
        i = 0
        print("total reward = ", round(total_reward, 1))
        total_reward = 0
        env.close()
    else:
        i += 1 

#%% Best AI Run 

env = car_environment()
state = env.reset()

AIActionList = [2, 3, 3, 3, 3, 1, 1, 3, 3, 3, 2, 1, 3, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 1, 3, 3, 1, 4, 0, 2, 0, 2, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 1, 4, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 3, 2, 4, 4, 1, 3, 1, 3, 3, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 2, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 3, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 3, 3, 2, 3, 3, 2, 3, 2, 3, 3, 3, 2, 3, 2, 3, 4, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 4, 4, 3, 3, 3, 3, 1, 2, 3, 2, 2, 2, 2, 2, 2, 3, 3, 3, 1, 3, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 1, 4, 1, 4, 2, 2, 0, 2, 1, 0, 2, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 2, 2, 3, 0, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 4, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 3, 1, 3, 2, 2, 2, 2, 2, 2, 3, 2, 4, 4, 1, 4, 1, 3, 1, 3, 1, 4, 2, 2, 2, 0, 0, 0, 2, 4, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 2, 2, 2, 4, 3, 3, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 4, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 3, 1, 3, 1, 3, 1, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 1, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 3, 3, 2, 2, 2, 2, 4, 2, 2, 3, 1, 3, 1, 4, 4, 2, 2, 0, 0, 0, 2, 0, 0, 4, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

human_control = False
i = 0
total_reward = 0
robot_trajectory = []
while env.run and i < len(AIActionList):
    next_state, reward, done = env.step(AIActionList[i], human_control)
    total_reward += reward
    x, y = env.player_car.centerx, env.player_car.centery
    robot_trajectory.append([x,y])
    if done:
        print("Total number of frames :" , i)
        print("Total convertedtime :" , i/60, 's')
        i = 0
        print("total reward = ", round(total_reward, 1))
        total_reward = 0
        env.close()
    else:
        i += 1 
        
        
#%% Displaying trajectories

import pygame
from CarEnvironment import car_environment_trajectory
        
Visualisation = car_environment_trajectory(human_trajectory, robot_trajectory)
while Visualisation.run:
    Visualisation.draw()
    
#%% Random Movements
#%%
import numpy as np
from CarEnvironment import car_environment_blind
import os

random_action_list = []
for i in range(50000):
    random_action_list.append(np.random.randint(0,5))
    
total_reward = 0
    
env = car_environment_blind(0)
state = env.reset()
random_data = []
i = 0

while env.run and i < len(random_action_list):
    next_state, reward, done= env.step(random_action_list[i])
    total_reward += reward
    i += 1
    if done:
        random_data.append(total_reward)
        total_reward = 0
        print(i)

mean = np.mean(random_data)
print(mean)


#%% Human Control and data retrieving
#%%
from CarEnvironment import car_environment_player

player_data = [] 

#%%

env = car_environment_player()
state = env.reset()
human_control = True
total_reward_simple = 0
total_reward_distance = 0
step = 0
position_list = []
while env.run:
    step += 1
    reward_simple, reward_distance, done, finish, pos = env.step(None, human_control)
    total_reward_simple += reward_simple
    total_reward_distance += reward_distance
    position_list.append(pos)
    if done:
        print(step/env.FPS)
        player_data.append([total_reward_simple, total_reward_distance, step/env.FPS, finish, position_list])
        step = 0
        total_reward_simple = 0
        total_reward_distance = 0 
        position_list = []
        
player_data_rewards = []
minimum = 1000000

for e in player_data:
    if e[3]:
        player_data_rewards.append(e[:3])
        if minimum >= e[2]:
            minimum = e[2]
            player_data_traj = e[4]

print(minimum)
# print(player_data_traj)


#%% Save Data

directory = "Hocher"
    
# Parent Directory path 
parent_dir = "Players_Data"
    
# Path 
path = os.path.join(parent_dir, directory) 

os.mkdir(path) 
print("Directory '% s' created" % directory) 
    
path = parent_dir + "/" + directory + "/"
        
# Saving data:
with open(path + 'rewards_data.txt', 'w') as f:
    for e in player_data_rewards:
        f.write(str(e) + '\n')
        
print("Done saving data: " + directory) 
        
# Saving data:
with open(path + 'Trajectory.txt', 'w') as f:
    for e in player_data_traj:
        f.write(str(e) + '\n')
        
print("Done saving trajectory: " + directory)


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

#%% 

trajectory = extract_trajectory_txt(path)

print(trajectory)

Visualisation = car_environment_trajectory(trajectory, robot_trajectory)
while Visualisation.run:
    Visualisation.draw()
