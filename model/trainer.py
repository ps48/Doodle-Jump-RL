import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class QTrainer:
    def __init__(self, model, lr, gamma, device):
        super(QTrainer, self). __init__()
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = device
        self.model.to(self.device)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        # (n, x)
        if state.shape[0] == 1:
            # (1, x) if short training
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        self.model.eval()
        with torch.no_grad():
            next_pred = self.model(next_state)
        
        # 1: predicted Q values with current state
        self.model.train()
        pred = self.model(state)
        
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        target = pred.clone().detach()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if (not done[idx]) and (idx != len(done)-1):
                Q_new = reward[idx] + self.gamma * torch.max(next_pred[idx])
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            del Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return float(loss)
