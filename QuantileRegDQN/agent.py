
import torch
import random
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from network import QDQN
from memory import ReplayBuffer
import torch.autograd as autograd


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR = 0.0001
UPDATE_FREQUENCY = 4
TRANSFER_RATE = 0.001
GAMMA = 0.99

class Agent():

    def __init__(self, state_space, action_space, memory_size=1000000, batch_size=32, seed=0, q_size=51):

        self.state_space = state_space
        self.action_space = action_space
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.seed = seed
        self.q_size = q_size

        self.current_model = QDQN(self.state_space, self.action_space, n_quantiles=self.q_size).to(device)
        self.target_model = QDQN(self.state_space, self.action_space, n_quantiles=self.q_size).to(device)
        self.optimizer = Adam(self.current_model.parameters(), lr=LR)

        self.memory = ReplayBuffer(self.action_space, self.memory_size, self.batch_size, self.seed)
        self.update_every = 0

        self.tau = (torch.Tensor((2 * np.arange(self.current_model.n_quantiles) + 1) / (2.0 * self.current_model.n_quantiles)).view(1, -1)).to(device)


    def soft_update(self, local_model, target_model, TRANSFER_RATE):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TRANSFER_RATE * local_param.data + (1.0 - TRANSFER_RATE) * target_param.data)

    def act(self, state, epsilon):

        if random.random() <= epsilon:
            action = random.choice(np.arange(self.action_space))
        else:
            action = self.current_model.act(state).cpu().numpy()
            #action = self.current_model.act(state, epsilon).cpu().numpy()
        return action

    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

        self.update_every += 1
        if self.update_every % UPDATE_FREQUENCY == 0:
            if len(self.memory) >= self.batch_size:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def learn(self, experience, gamma):

        sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_done = experience



        #print(self.current_model(sampled_state).shape)
        #print(self.current_model(sampled_state)[0:self.batch_size, 0: self.action_space])
        #print(self.current_model(sampled_state))
        #print(self.current_model(sampled_state).shape)

        #print(sampled_action.shape)
        #print(sampled_action.expand(self.batch_size, self.q_size))
        #print(sampled_action.unsqueeze(1).expand(self.batch_size, 1, self.q_size).shape)
        action = sampled_action.unsqueeze(1).expand(self.batch_size, 1, self.q_size)

        #print(self.current_model(sampled_state))
        #print(self.current_model(sampled_state).gather(1, action).squeeze(1))



        theta = self.current_model(sampled_state).gather(1, action).squeeze(1)
        #theta = self.current_model(sampled_state).mean(2)

        z_next = self.target_model(sampled_next_state).detach()
        #print(z_next)
        #print(z_next.shape)

        z_next_max = z_next[np.arange(self.batch_size), z_next.mean(2).max(1)[1]]
        #print(z_next_max)
        Ttheta = sampled_reward + GAMMA * (1 - sampled_done) * z_next_max
        #print(Ttheta)
        #print(Ttheta.shape)
        #print(theta.shape)
        diff = Ttheta.t().unsqueeze(-1) - theta

        loss = self.huber(diff) * (self.tau - (diff.detach() < 0).float()).abs()
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.current_model, self.target_model, TRANSFER_RATE)

    def huber(self, x, k=1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))




