import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

#import the policy model
from model import Policy

#import options
from options import options
options = options()
opts = options.parse()

#import util module
from util import render_text_envq

import torch
torch.manual_seed(0) # set random seed
import torch.optim as optim

env = gym.make(opts.env)
env.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_size=env.observation_space.shape[0]
policy = Policy(s_size=state_size, h_size=opts.hidden).to(device)
if opts.print_model:
    print(policy)
optimizer = optim.Adam(policy.parameters(), lr=opts.lr)

def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= opts.win_cond:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores

scores = reinforce(n_episodes=opts.num_episodes, max_t=opts.max_iteration, gamma=opts.discount_rate, print_every=opts.print_every)

if opts.render:
    render_text_envq(env, policy)
