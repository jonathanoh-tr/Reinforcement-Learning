
import torch
import random
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from network import QDQN
from SumTree import SumTree
import torch.autograd as autograd


USE_CUDA = not torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

LR = 0.0001
UPDATE_FREQUENCY = 4
TRANSFER_RATE = 0.001
GAMMA = 0.99

SMALL_EPSILON = 0.01
alpha = 0.4

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

        self.SumTree = SumTree(capacity=self.memory_size)

        self.update_every = 0

        self.tau = (torch.Tensor((2 * np.arange(self.current_model.n_quantiles) + 1) / (2.0 * self.current_model.n_quantiles)).view(1, -1)).to(device)


    def soft_update(self, local_model, target_model, TRANSFER_RATE):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TRANSFER_RATE * local_param.data + (1.0 - TRANSFER_RATE) * target_param.data)

    def act(self, state, epsilon):

        #if random.random() <= epsilon:
            #action = random.choice(np.arange(self.action_space))
        #else:
        action = self.current_model.act(state).cpu().numpy()

        return action

    def step(self, state, action, reward, next_state, done):

        data = [state, action, reward, next_state, done]
        self.SumTree.add(self.SumTree.get_max_p() ** alpha, data)

        self.update_every += 1
        if self.update_every % UPDATE_FREQUENCY == 0:
            if self.SumTree.write >= self.batch_size:

                experience = self.SumTree.sample(self.batch_size -1)

                self.learn(experience, GAMMA)

    def learn(self, experience, gamma):

        sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_done, index, weight = experience

        #print(sampled_action.shape)
        action = sampled_action.unsqueeze(1).expand(self.batch_size, 1, self.q_size)

        theta = self.current_model(sampled_state).gather(1, action).squeeze(1)

        z_next = self.target_model(sampled_next_state).detach()


        z_next_max = z_next[np.arange(self.batch_size), z_next.mean(2).max(1)[1]]

        Ttheta = sampled_reward + GAMMA * (1 - sampled_done) * z_next_max

        diff = Ttheta.t().unsqueeze(-1) - theta

        loss = self.huber(diff) * (self.tau - (diff.detach() < 0).float()).abs()
        loss = loss.mean(0).mean(1)

        #print(loss)
        #print(loss.shape)

        for i in range(len(loss) -1):
            delta = loss[i].detach()
            self.SumTree.update(index[i], (delta + SMALL_EPSILON) ** alpha)

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.current_model, self.target_model, TRANSFER_RATE)

        self.current_model.reset_noise()
        self.target_model.reset_noise()

    def huber(self, x, k=1.0):
        return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))




