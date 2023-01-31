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

### Agent actions

![image](https://user-images.githubusercontent.com/66775006/215754247-c332e252-9fcd-4a60-b766-27969d8f32ac.png)



### Input of the Neural Network


![IMe](https://user-images.githubusercontent.com/66775006/215752818-6acd229d-3782-4e98-9da1-8c39b33325df.jpg)


### Rewards

## Multi Agent
### Figure of the system
### Saving and reloading a model

```
directory = "Model_600"
    
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
print("Done saving model: " + directory)

```

## Results

![image](https://user-images.githubusercontent.com/66775006/215592124-2b5531ca-5a56-460c-821a-2c71b11efd92.png)

![image](https://user-images.githubusercontent.com/66775006/215592143-03864f67-eeb6-4b3d-9f70-5ad04cf728c9.png)

![image](Resources/Animation.gif)

## Conclusion of the project


