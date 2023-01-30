# :red_car: Reinforcement Learning in a 2D Racing Game

In this project we are going to implement a Deep Reinforcement Learning algorithm in a 2D Racing car game.
The goal of the project is to be able to build and train an autonomous AI that is capable to drive along the track. 

## Using the code

You can use the jupyter version of the project with the file ```AsynchronousDoubleDeepQLearning.ipynb```.
The code is quite long but is splitted in several classes that I will explain in this file.

Other useful documents can be found in the **Resources** Folder.

## Car Environment
Credits for the Environment: 

GitHub: https://github.com/techwithtim/Pygame-Car-Racer

Youtube: https://www.youtube.com/watch?v=L3ktUWfAMPg&list=PLzMcBGfZo4-kmY7Nh4kI9kPPnxJ5JMRPj&index=2

### Car

```
class PlayerCar:

    def __init__(self, max_vel, rotation_vel, track_mask):

    def move(self):
    
    def handle_collision(self):

    def reset(self):
       
    def Inputs(self, cluster_angle):
```

### Track

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
### Agent actions
### Input of the Neural Network


![IMe](https://user-images.githubusercontent.com/66775006/215596598-0c64ed73-0da3-40f0-a595-720f1e53fb86.jpg)


### Rewards

## Multi Agent
### Figure of the system
### Saving and reloading a model

## Results

![image](https://user-images.githubusercontent.com/66775006/215592124-2b5531ca-5a56-460c-821a-2c71b11efd92.png)

![image](https://user-images.githubusercontent.com/66775006/215592143-03864f67-eeb6-4b3d-9f70-5ad04cf728c9.png)

![image](Resources/Animation.gif)

## Conclusion of the project


