# :red_car: Reinforcement Learning in a 2D Racing Game

In this project we are going to implement a Deep Reinforcement Learning algorithm in a 2D Racing car game.
The goal of the project is to be able to build and train an autonomous AI that is capable to drive along the track. 
We are going to focus on the Q-Learning Algorithm because of its simplicity. 

## Using the code

You can use the jupyter version of the project with the file ```AsynchronousDoubleDeepQLearning.ipynb```.
The code is quite long but is splitted in several classes that I will explain in this file.

Other useful documents can be found in the **Resources** folder.

The **Model** folder contains trained model that achieved good results in regard of our problem.

## Explication of Reinforcement Learning

Reinforcement Learning is the third class of problems in Machine Learning: we have supervised learning (classification, regression), unsupervised learning (autoencoder, PCA), and reinforcement learning. The reinforcement learning involves an environment and an agent. The agent performs an action `A` , gets rewarded with reward `R`, and changes to anoter state `S`. This concept is explained in the following diagramm.

![image](https://user-images.githubusercontent.com/66775006/215762340-583f3da4-83fa-4e6a-b76f-f282b253cfb2.png)

## Car Environment

First of all, we need an environment to work with. The environment in Reinforcement Learning contains all information needed for the agent to learn. 
Pratically, an environment stores its own state. The change of state happends when calling the `step()` method of the environment with changes the previous state to a next state according to the internal rules of the environment. A reinforcement learning environment canonicaly returns a tuple of information for the agent. Like the environment made by OpenAI Gym (https://github.com/openai/gym), our environement will return the following information for every call to the `step()` method:

- **observation**: the state of the environment before taking an action 
- **action**: the action taken 
- **reward**: the reward that the agent got in regard of its *(action, state)* pair
- **new_observation**: the new state returned by the environment in consequence of the action
- **done**: a boolean indicating if the new state is terminal or not. 

Additionnaly an environment should be equiped with a `reset()` method which allows for reseting its state towards the default state. This is very usefull when an episode terminates. An episode is a serie of steps from the environment until crosses a terminal state.

For our environment, we used a modified version of a race car game that I found on GitHub, here are the link to this project.

GitHub: https://github.com/techwithtim/Pygame-Car-Racer

Youtube: https://www.youtube.com/watch?v=L3ktUWfAMPg&list=PLzMcBGfZo4-kmY7Nh4kI9kPPnxJ5JMRPj&index=2

Our environment features a Graphical User Interface (GUI) and helps us visualize the result of our training on our agent. The GUI is displayed on the following picture.

![Circuit ](https://user-images.githubusercontent.com/66775006/215759383-94b139d4-76b7-4b7f-a0f6-51e71eac0a16.png)

One can see a car in a racetrack that is subdivided in several colored boxes. Each of the boxes corresponds to a type of racetrack part. For example example, red parts correspond to straight parts while purple parts correspond to left turns. 

For simplicity and readability, our environment uses two classes: the `PlayerCar` and the `car_environment`. The simplified code is displayed in the following part:

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

Now that we have our environment, we can start the code of the agent. The principle of Q-Learning is quite simple: at every step the agent compute the *q-value* of all the possible actions, which correponds to value of each choice. Then, the agent takes the best choice if it wants to maximize its immediate reward, or takes another one if it has to explore more possibilities. We are also using a neural network as a function approximator (this is called DQN for Deep Q Network). The diagramm of DQN is displayed bellow:

![Q-Learning](https://user-images.githubusercontent.com/66775006/215767109-4a94e70b-745b-419b-8de6-5f3c0d7f4faf.png)

When updating the weights of our model, we use the Bellman equation as follow:

![Equation](https://user-images.githubusercontent.com/66775006/215766980-c8ea0673-de2d-4a1f-95de-82d8a6d92f5e.png)

Before we display the code of our Agent, I have to underline the tricks that are used in this version of DQN and that are often cited in research papers:
- **Epsilon-greedy strategy**: as explained before, exploration of the environment is a key element and this allows for random exploration with a probability epsilon
- **Double Networks**: two neural networs are used in order to improve stability of learning. 
- **Experience replay**: experience from past steps is stored in the memory and used for fiting the model with stochastic gradient descent.

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

We are now going to explain the different actions possible for the agent. As for many race games, the car is fully controllable with the direction keys:

![keys](https://user-images.githubusercontent.com/66775006/215758287-b4b956c2-6d9c-40a9-932b-a46cf5ab7c2e.png)

All actions possible are thus combinations of those actions: for instance pressing the left key and the forward keys makes the car both increase its velocity and turn to the left.


### Input of the Neural Network

For inputs of the neural network, we used a very simple yet effective method called Lidar. This methods uses raycasting in several directions and gets the distances between the circuit limits and the car for each corresponding direction. A visualisation of this technique is shown in the following pictures, when one can see the car and the different laser beams represented in red color.

![IMe](https://user-images.githubusercontent.com/66775006/215752818-6acd229d-3782-4e98-9da1-8c39b33325df.jpg)

### Rewards

Now that we have the environment, the actions for the agent and the inputs of the model, we need a way to tell the agent if its action are rewarded in regard of the current state. 

We choose here to be very simple and to assign the reward to be proportional to the distanc travelled in the right direction. The angle of the right direction depends on the part of the track the car is in, thanks to the `checkCluster()` method.

The part of the code that deals with the speed reward is written just below:

```
def rewardSpeed(self):
        roadAngle = self.checkCluster()
        radians = math.radians(self.player_car.angle - roadAngle)
        reward = self.player_car.vel*math.cos(radians)/self.player_car.max_vel
        return reward
```

## Multi Agent

```
class MultiDQN_Agent():

    def __init__(self, env):
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.05
        self.decay = 0.003
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Main Model (updated every 4 steps)
        self.model = self.createModel(self.observation_space, self.action_space)
        # Target Model (updated every 100 steps)
        self.target_model = self.createModel(self.observation_space, self.action_space)
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
        
        self.n_env = 10
        
        self.totalrewards = [] # Contains all done bool and cumulative rewards from environments
        self.rewardsData = [] # Data from all past total cumulative rewards for data analysis
        
        self.states_and_env = []
        self.init_states_and_env()
        
    def init_states_and_env(self):
        for i in range(self.n_env):
            blind_env = car_environment_blind(i)
            state = blind_env.reset()
            self.states_and_env.append([state, blind_env])
            self.totalrewards.append([False, 0])
        
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
    
    def getAction(self):    
        random_env_list = []
        greedy_env_list = []
        experiences = []
    
        for index in range(self.n_env):
            random_number = np.random.rand()
            if random_number <= self.epsilon:
                # Explore
                random_env_list.append(self.states_and_env[index])
            else:
                # Exploit best known action
                greedy_env_list.append(self.states_and_env[index])
        
        if len(greedy_env_list) >= 1:
            
            if len(greedy_env_list) == 1:
                predicted = self.model.predict([greedy_env_list[0][0]])
            else:
                states = []
                for index, (state, env) in enumerate(greedy_env_list):
                    states.append(state)
                predicted = self.model.predict(states)
                
            for index, (state, env) in enumerate(greedy_env_list):
                action = np.argmax(predicted[index])
                next_state, reward, done = greedy_env_list[index][1].step(action)
                reward_env = self.totalrewards[env.index][1] + reward
                self.totalrewards[env.index] = [done, reward_env]
                state = greedy_env_list[index][0]
                experiences.append([state, action, reward, next_state, done])

                greedy_env_list[index][0] = next_state
            
        if len(random_env_list) >= 1:
            actions = [np.random.randint(self.action_space) for i in range(len(random_env_list))]
            for index, (state, env) in enumerate(random_env_list):
                action = actions[index]
                next_state, reward, done = random_env_list[index][1].step(action)
                reward_env = self.totalrewards[env.index][1] + reward
                self.totalrewards[env.index] = [done, reward_env]
                state = random_env_list[index][0]
                experiences.append([state, action, reward, next_state, done])

                random_env_list[index][0] = next_state
            
        self.states_and_env = random_env_list + greedy_env_list
        
        for index, (done, total_reward) in enumerate(self.totalrewards): 
            if done:
                self.episode += 1
                print('Episode = {}, total reward: {}, , epsilon = {}'.format(self.episode, \
                                  round(total_reward, 1), round(self.epsilon, 3)))
                self.totalrewards[index] = [False, 0]
                self.rewardsData.append(total_reward) # total training and done
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.episode)
            
        return experiences
        
        
    def train(self, experiences):
        
        for experience in experiences:
        
            observation, action, reward, new_observation, done = experience
            self.replay_memory.append([observation, action, reward, new_observation, done])

            self.steps_to_update_target_model += 1

            if self.steps_to_update_target_model >= 100 and done:
                self.target_model.set_weights(self.model.get_weights())
                self.steps_to_update_target_model = 0

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


