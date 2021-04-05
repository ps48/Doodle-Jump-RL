import numpy as np
import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)

    return discounted[::-1]


def process_memory(memory, device, gamma=0.99, discount_rewards=True):
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []
    for action, reward, state, next_state, done in memory:
        if action is not None:
            actions.append(action)
        else:
            actions.append((np.zeros((1, 3))))
        if reward is not None:
            rewards.append(reward)
        else:
            rewards.append(0)
        if state is not None:
            states.append(state)
        else:
            states.append(np.zeros((1, 80, 80)))
        if next_state is not None:
            next_states.append(next_state)
        else:
            next_states.append(np.zeros((1, 80, 80)))
        if done is not None:
            dones.append(done)
        else:
            dones.append(False)

    if discount_rewards:
        # if False and dones[-1] == 0:
        #     rewards = discounted_rewards(rewards + [last_value], dones + [0], gamma)[:-1]
        # else:
        rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(actions).to(device)
    states = t(states).to(device)
    next_states = t(next_states).to(device)
    rewards = t(rewards).view(-1, 1).to(device)
    dones = t(dones).view(-1, 1).to(device)
    return actions, rewards, states, next_states, dones


def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


###########

class A2CLearner():
    def __init__(self, actor, critic, device, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.device = device

    def learn(self, memory, steps, writer, discount_rewards=True):
        actions, rewards, states, next_states, dones = process_memory(memory, self.device, self.gamma, discount_rewards)

        if discount_rewards:
            td_target = rewards
        else:
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        value = self.critic(states)
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()

        actor_loss = (-logs_probs * advantage.detach()).mean() - entropy * self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()

        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        writer.add_histogram("gradients/actor",
                             torch.cat([p.grad.view(-1) for p in self.actor.parameters()]), global_step=steps)
        writer.add_histogram("parameters/actor",
                             torch.cat([p.data.view(-1) for p in self.actor.parameters()]), global_step=steps)
        self.actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        writer.add_histogram("gradients/critic",
                             torch.cat([p.grad.view(-1) for p in self.critic.parameters()]), global_step=steps)
        writer.add_histogram("parameters/critic",
                             torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=steps)
        self.critic_optim.step()

        # reports
        writer.add_scalar("losses/log_probs", -logs_probs.mean(), global_step=steps)
        writer.add_scalar("losses/entropy", entropy, global_step=steps)
        writer.add_scalar("losses/entropy_beta", self.entropy_beta, global_step=steps)
        writer.add_scalar("losses/actor", actor_loss, global_step=steps)
        writer.add_scalar("losses/advantage", advantage.mean(), global_step=steps)
        writer.add_scalar("losses/critic", critic_loss, global_step=steps)