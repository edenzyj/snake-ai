import time
import torch
import tensorflow as tf
import keras
from keras import layers
import random
import numpy as np
from collections import deque

from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, w=640, h=480):
        self.n_games = 0
        self.epsilon = 1.0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.reward = 0
        
        '''# Q-Learning TD
        self.alpha = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = {}'''
        
        # Q-Learning tensorflow
        #self.model = self.network()
        
        # DQN
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def network(self, weights=None):
        model = keras.Sequential([
            layers.Dense(120, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(120, activation='relu'),
            layers.Dropout(0.15),
            layers.Dense(3, activation='softmax'),
        ])
        
        loss_func = keras.losses.mean_squared_error
        opt = keras.optimizers.Adam(learning_rate=LR)
        
        model.compile(loss=loss_func, optimizer=opt)
        
        return model

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def get_q_value(self, state, move):
        snake_state = tuple(i for i in state)
        if type(move) == int:
            action = move
        else:
            action = 0
            for i in range(3):
                if move[i] == 1:
                    action = i
        return self.q_table.get((snake_state, action), 0.0)
    
    def update_long_q_value(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        for state, move, reward, next_state, done in mini_sample:
            old_value = self.get_q_value(state, move)
            next_max = max(self.get_q_value(next_state, a) for a in range(3))
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
            snake_state = tuple(i for i in state)
            action = 0
            for i in range(3):
                if move[i] == 1:
                    action = i
            self.q_table[(snake_state, action)] = new_value
    
    def update_short_q_value(self, state, move, reward, next_state, done):
        old_value = self.get_q_value(state, move)
        next_max = max(self.get_q_value(next_state, a) for a in range(3))
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        snake_state = tuple(i for i in state)
        action = 0
        for i in range(3):
            if move[i] == 1:
                action = i
        self.q_table[(snake_state, action)] = new_value
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        '''# Q-Learning tensorflow
        for state, action, reward, next_state, done in mini_sample:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        '''
        # DQN
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        '''# Q-Learning tensorflow
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
        target_f = self.model.predict(np.array([state]))
        target_f[0][np.argmax(action)] = target
        self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        '''
        # DQN
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        
        final_move = [0,0,0]
        
        '''# Q-Learning TD
        if random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            q_values = [self.get_q_value(state, a) for a in range(3)]
            move = np.argmax(q_values)
            final_move[move] = 1
        '''    
        
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            '''# Q-Learning tensorflow
            state0 = tf.convert_to_tensor(state, dtype=tf.float32)
            state0 = state0[tf.newaxis, :]
            prediction = self.model.predict(state0)[0]
            move = 2
            for i in range(2):
                if prediction[i] > prediction[move]:
                    move = i
            '''    
            # DQN
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        try:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # update short time q-value of TD method
            #agent.update_short_q_value(state_old, final_move, reward, state_new, done)
            
            # train short memory of other methods
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                
                # update long time q-value of TD method 
                #agent.update_long_q_value()
                
                # train long memory of other methods
                agent.train_long_memory()

                if score > record:
                    record = score
                    # save model of other methods
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                
                # Q-Learning TD
                #agent.decay_epsilon()
                
        except KeyboardInterrupt:
            time.sleep(10)

if __name__ == '__main__':
    train()
