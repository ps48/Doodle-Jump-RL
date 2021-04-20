import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class QTrainer:
    def __init__(self, model, lr, gamma, device, num_channels, attack_eps):
        super(QTrainer, self). __init__()
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = device
        self.model.to(self.device)
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std = (0.229, 0.224, 0.225)
        
        if num_channels == 1:
            imagenet_mean = np.mean(imagenet_mean)
            imagenet_std = np.mean(imagenet_std)
            
        self.mu = torch.tensor(imagenet_mean).view(num_channels, 1, 1).cuda()
        self.std = torch.tensor(imagenet_std).view(num_channels, 1, 1).cuda()
        self.upper_limit = ((1 - self.mu)/ self.std)
        self.lower_limit = ((0 - self.mu)/ self.std)
        self.attack_eps = attack_eps / self.std
        self.attack_step = (1.25*attack_eps) / self.std
                
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
    
    def clamp(self, X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)
    
    def create_adv_state(self, state, action, reward, next_state, done):
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
        
        delta = torch.zeros_like(state).to(self.device)
        for j in range(len(self.attack_eps)):
            delta[:, j, :, :].uniform_(-self.attack_eps[j][0][0].item(), self.attack_eps[j][0][0].item())
        delta.data = self.clamp(delta, self.lower_limit - state, self.upper_limit - state)
        delta.requires_grad = True
        self.model.train()
        output = self.model((state + delta[:state.size(0)]).float())
        
        # 1: predicted Q values with current state
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
        loss = self.criterion(output, target)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = self.clamp(delta + self.attack_step * torch.sign(grad), -self.attack_eps, self.attack_eps)
        delta.data[:state.size(0)] = self.clamp(delta[:state.size(0)], self.lower_limit - state, self.upper_limit - state)
        delta = delta.detach()
        return delta[:state.size(0)]