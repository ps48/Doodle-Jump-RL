import os
import argparse
import random
import numpy as np
import cv2
import torch
from collections import deque
from game.doodlejump import DoodleJump
from model import Deep_QNet, Deep_RQNet, QTrainer
from helper import plot


class Agent:
    def __init__(self, args):
        self.n_games = 0
        self.epsilon = 0
        self.ctr = 1
        self.store_frames = args.store_frames
        self.image_h = args.height
        self.image_w = args.width
        self.memory = deque(maxlen=args.max_memory)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        if args.model=="dqn":
            self.model = Deep_QNet()
        elif args.model=="drqn":
            self.model = Deep_RQNet()
        self.trainer = QTrainer(model=self.model, lr=self.lr, gamma=self.gamma, device=self.device)

    def get_state(self, game):
        state = game.getCurrentFrame()
        img = cv2.cvtColor(cv2.resize(state, (self.image_w, self.image_h)), cv2.COLOR_BGR2GRAY)
        M = cv2.getRotationMatrix2D((self.image_w / 2, self.image_h / 2), 270, 1.0)
        img = cv2.warpAffine(img, M, (self.image_h, self.image_w))

        if self.store_frames:
            os.makedirs("./image_dump", exist_ok=True)
            cv2.imwrite("./image_dump/"+str(self.ctr)+".jpg", img)
            self.ctr+=1
        state = np.expand_dims(img, axis=0)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

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


def train(game, args):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK']='True'

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(args)
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
    parser = argparse.ArgumentParser(description='RL Agent for Doodle Jump')
    parser.add_argument("--macos", action="store_true", help="select model to train the agent")
    parser.add_argument("--human", action="store_true", help="playing the game manually without agent")
    parser.add_argument("-d", "--difficulty", type=str, default="EASY", choices=["EASY", "MEDIUM", "HARD"], help="select difficulty of the game")
    parser.add_argument("-m", "--model", type=str, default="dqn", choices=["dqn", "drqn"], help="select model to train the agent")
    parser.add_argument("-w", "--weights", type=str, help="path to weights of an earlier trained model")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="set learning rate for training the model")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help="set discount factor for q learning")
    parser.add_argument("--max_memory", type=int, default=10000, help="Buffer memory size for long training")
    parser.add_argument("--store_frames", action="store_true", help="store frames encountered during game play by agent")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for long training")
    parser.add_argument("--height", type=int, default=80, help="set the image height post resize")
    parser.add_argument("--width", type=int, default=80, help="set the image width post resize")

    args = parser.parse_args()
    # can pass in 'EASY', 'MEDIUM', 'DIFFICULT' in the constructor. default is EASY.
    game = DoodleJump(difficulty=args.difficulty)
    if args.human:
        game.run()
    else:
        train(game, args)
