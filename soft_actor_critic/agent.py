
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from model import Value, soft_Q, Policy
from memory import ReplayBuffer

TAU = 1e-2
BUFFERSIZE = 1000000
class Agent():

    def __init__(self, state_size, action_size, batch_size=128, gamma=0.99, mean_lambda=1e-3,std_lambda=1e-3,z_lambda=0.0):

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = ReplayBuffer(BUFFERSIZE, self.batch_size)

        self.mean_lambda = mean_lambda
        self.std_lambda = std_lambda
        self.z_lambda = z_lambda

        self.current_value  = Value(state_size).to(device)
        self.target_value   = Value(state_size).to(device)

        self.softQ = soft_Q(state_size, action_size)
        self.policy = Policy(state_size,action_size)

        self.value_optimizer = optim.Adam(self.current_value.parameters(), lr=3e-4)
        self.soft_q_optimizer = optim.Adam(self.softQ.parameters(), lr=3e-4)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

    def act(self, state):

        #state = torch.from_numpy(np.asarray(state)).float().to(device)
        action = self.policy.act(state)

        if self.memory.__len__() > self.batch_size:
            self.update()

        return action

    def add_to_memory(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

    def update(self):

        state, action, reward, next_state, done = self.memory.sample()

        expected_soft_q_value = self.softQ.forward(state, action)
        expected_value = self.current_value.forward(state)

        new_action, log_prob, z, mean, log_std = self.policy.evaluate(state)

        target_value = self.target_value.forward(next_state)
        next_soft_q_value = reward + self.gamma * target_value * (1 - done)

        q_val_mse = F.mse_loss(expected_soft_q_value, next_soft_q_value.detach())

        expected_new_q_val = self.softQ.forward(state, new_action)
        next_value = expected_new_q_val - log_prob
        val_loss = F.mse_loss(expected_value, next_value.detach())

        log_prob_target = expected_new_q_val - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = self.mean_lambda * mean.pow(2).mean()
        std_loss = self.std_lambda * log_std.pow(2).mean()
        z_loss = self.z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_val_mse.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        val_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.soft_update(self.current_value, self.target_value, TAU)

    def soft_update(self, local_model, target_model, TRANSFER_RATE):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TRANSFER_RATE * local_param.data + (1.0 - TRANSFER_RATE) * target_param.data)

