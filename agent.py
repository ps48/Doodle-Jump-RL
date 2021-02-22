import torch
import random
import numpy as np
import cv2
from collections import deque
import torch
from game.doodlejump import DoodleJump
from model import Deep_QNet, Deep_Recurrent_QNet, QTrainer
from helper import plot

# These 2 lines of code are for the code to run on mac. Facing some issues due to duplicate openMP libraries. Ignore these.
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''

MAX_MEMORY = 10000
IMAGE_H = 80
IMAGE_W = 80
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen = MAX_MEMORY)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # For DRQN self.model = Deep_Recurrent_QNet()
        # For DQN self.model = Deep_QNet()
        
        # self.model = Deep_QNet() #input_size = [1,4,80,80], output_size = 80)
        self.model = Deep_Recurrent_QNet()
        
        self.lr = LR
        self.trainer = QTrainer(model = self.model, lr=self.lr, gamma=self.gamma, device=self.device)
        self.ctr = 1


    def get_state(self, game):
        state = game.getCurrentFrame()
        img = cv2.cvtColor(cv2.resize(state, (IMAGE_W, IMAGE_H)), cv2.COLOR_BGR2GRAY)
        M = cv2.getRotationMatrix2D((IMAGE_W / 2, IMAGE_H / 2), 270, 1.0)
        img = cv2.warpAffine(img, M, (IMAGE_H, IMAGE_W))
        # NOTE: Uncomment to store images
        # cv2.imwrite("image_dump/"+str(self.ctr)+".jpg", img)
        # self.ctr+=1
        state = np.expand_dims(img, axis=0)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = DoodleJump() # can pass in 'EASY', 'MEDIUM', 'DIFFICULT' in the constructor. default is EASY.
    print("Now playing")
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.playStep(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, [done])

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.gameReboot()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()
