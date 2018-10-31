
import torch
import random
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from model import Actor, Critic
from replay import ReplayBuffer
import torch.autograd as autograd


USE_CUDA = not torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

LR = 0.0001
UPDATE_FREQUENCY = 4
TRANSFER_RATE = 0.001
GAMMA = 0.99
WEIGHT_DECAY = 0.0001

class Agent():

    def __init__(self, state_space, action_space, memory_size=1000000, batch_size=32, seed=0, q_size=51):

        self.state_space = state_space
        self.action_space = action_space
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.seed = seed
        self.q_size = q_size

        self.current_model_actor = Actor(self.state_space, self.action_space, seed=self.seed).to(device)
        self.target_model_actor = Actor(self.state_space, self.action_space, seed=self.seed).to(device)
        self.actor_optimizer = Adam(self.current_model_actor.parameters(), lr=LR)

        self.current_model_critic = Critic(self.state_space, self.action_space, seed=self.seed).to(device)
        self.target_model_critic = Critic(self.state_space, self.action_space, seed=self.seed).to(device)
        self.critic_optimizer = Adam(self.current_model_critic.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        self.memory = ReplayBuffer(self.action_space, self.memory_size, self.batch_size, self.seed)
        self.update_every = 0

        self.noise = OUNoise(self.action_space, self.seed)

    def soft_update(self, local_model, target_model, TRANSFER_RATE):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TRANSFER_RATE * local_param.data + (1.0 - TRANSFER_RATE) * target_param.data)

    def act(self, state, noise=True):

        if noise:
            action = self.current_model_actor(state) + self.noise.sample()
        else:
            action = self.current_model_actor(state)
            #action = self.current_model.act(state, epsilon).cpu().numpy()
        return action.cpu().numpy()

    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)

        self.update_every += 1
        if self.update_every % UPDATE_FREQUENCY == 0:
            if len(self.memory) >= self.batch_size:
                experience = self.memory.sample()
                self.learn(experience, GAMMA)

    def reset(self):
        self.noise.reset()

    def learn(self, experience, gamma):

        sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_done = experience

        actions_next = self.target_model_actor(sampled_next_state)
        Q_target_next = self.target_model_critic(sampled_next_state, actions_next)

        Q_target = sampled_reward + (gamma * Q_target_next * (1- sampled_done))

        Q_expected = self.current_model_critic(sampled_state, sampled_action)

        loss = F.mse_loss(Q_expected, Q_target)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.current_model_critic(sampled_state)
        actor_loss = -self.current_model_critic(sampled_state, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # ------------------- update target network ------------------- #
        self.soft_update(self.current_model_actor, self.target_model_actor, TRANSFER_RATE)
        self.soft_update(self.current_model_critic, self.target_model_critic, TRANSFER_RATE)



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


