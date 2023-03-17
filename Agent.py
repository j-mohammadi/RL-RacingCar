import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import random
from CarEnvironment import car_environment_blind
import pandas as pd
import matplotlib.pyplot as plt

class Simple_DQN_Agent():
    
    def __init__(self, observation_space, action_space):
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.05
        self.decay = 0.005

        self.action_space = action_space
        # Main Model
        self.model = self.createModel(observation_space, action_space)

        self.replay_memory = deque(maxlen=50_000)
        
        self.learning_rate = 0.7 # Learning rate
        self.discount_factor = 0.618

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
        model.add(keras.layers.Input(shape=(state_shape,)))
        model.add(keras.layers.Dense(16, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(16, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=NNlearning_rate), metrics=['accuracy'])
        return model
    
    def getAction(self, state):
        random_number = np.random.rand()
        if random_number <= self.epsilon:
            # Explore
            action = np.random.randint(0,self.action_space)
        else:
            # Exploit best known action
            predicted = self.model.predict([state])
            action = np.argmax(predicted)
        return action
        
        
    def train(self, experience):
        
        observation, action, reward, new_observation, done = experience
        self.total_training_rewards += reward
        
        self.steps_to_update_target_model += 1
        
        if done:
            self.episode += 1
            self.total_training_rewards = 0
        
        if done:
            self.steps_to_update_target_model = 0
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.episode)
        
        current_qs_list = self.model.predict([observation])[0]
        future_qs_list = self.model.predict([new_observation])[0]

        X = []
        Y = []
        if not done:
            max_future_q = reward + self.discount_factor * np.max(future_qs_list)
        else:
            max_future_q = reward

        current_qs = current_qs_list
        current_qs[action] = (1 - self.learning_rate) * current_qs[action] + self.learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), verbose=0, shuffle=True)


class minDQN_Agent():
    
    def __init__(self, observation_space, action_space):
        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.05
        self.decay = 0.003

        self.action_space = action_space
        self.observation_space = observation_space
        # Main Model (updated every 4 steps)
        self.model = self.createModel()
        # Target Model (updated every 100 steps)
        self.target_model = self.createModel()
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
    
    def createModel(self):
        NNlearning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(16, input_shape=(self.observation_space,), activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(16, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(self.action_space, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=NNlearning_rate), metrics=['accuracy'])
        return model
    
    def getAction(self, state):
        random_number = np.random.rand()
        if random_number <= self.epsilon:
            # Explore
            action = np.random.randint(0,self.action_space)
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
        
        self.n_env = 20
        
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
        model.add(keras.layers.Dense(16, input_shape=(self.observation_space,), activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(16, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(self.action_space, activation='linear', kernel_initializer=init))
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
#                 print('Episode = {}, total reward: {}, , epsilon = {}'.format(self.episode, \
#                                   round(total_reward, 1), round(self.epsilon, 3)))
                self.totalrewards[index] = [False, 0]
                self.rewardsData.append(total_reward) # total training and done
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.episode)
            
        return experiences
        
    def display_progress(self):
        data = self.rewardsData
        weight = 30
        if len(data) > weight:
            numbers_series = pd.Series(data)
            windows = numbers_series.rolling(weight)
            moving_averages = windows.mean()
            moving_averages_list = moving_averages.tolist()

            plt.figure(figsize=(12, 4), dpi=80)
            plt.plot(moving_averages_list)
            plt.xlabel("Episode / Current epsilon: " + str(round(self.epsilon, 3)))
            plt.ylabel("Reward")
            plt.title("Average Reward over time")
            plt.grid(True)
            plt.show()
        
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

def extract_memory_txt(path):
    with open(path + 'memory.txt') as f:
        lines = f.readlines()

    memory_data = []
    for i in range(len(lines)):

        line = lines[i][2:-2]

        data = []
        state = []
        start = 0
        for i in range(len(line)):
            if line[i] == ']':
                state.append(float(line[start:i]))
                break
            elif line[i] == ',':
                state.append(float(line[start:i]))
                start = i+1

        data.append(state)
        line = line[i+3:]
        data.append(int(line[0]))
        line = line[3:]
        for i in range(len(line)):
            if line[i] == ',':
                data.append(float(line[:i]))
                break
        line = line[i+3:]

        state = []
        start = 0
        for i in range(len(line)):
            if line[i] == ']':
                state.append(float(line[start:i]))
                break
            elif line[i] == ',':
                state.append(float(line[start:i]))
                start = i+1
        data.append(state)
        line = line[i+3:]
        if line[0] == 'F':
            data.append(False)
        else:
            data.append(True)

        memory_data.append(data)
    return memory_data

def extract_reward_data(path):
    with open(path + 'rewards_data.txt') as f:
        lines = f.readlines()
    rewards_data = []
    for i in range(len(lines)):
        rewards_data.append(float(lines[i]))
    
    return rewards_data
    