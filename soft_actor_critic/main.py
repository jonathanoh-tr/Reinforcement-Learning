
import torch
import gym
import numpy as np
from collections import deque

from agent import Agent
from utility import Action_Normalizer

env = Action_Normalizer(gym.make("Pendulum-v0"))

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

policy = Agent(state_size, action_size)

n_epsiodes = 100
n_steps = 500
scores_deque = deque(maxlen=100)
print_every = 5
for i in range(n_epsiodes):
    state = env.reset()
    rewards = 0

    for steps in range(n_steps):

        action = policy.act(state)
        action = action.item()
        next_state, reward, done, _ = env.step(action)
        policy.add_to_memory(state, action, reward, next_state, done)

        state = next_state
        rewards += reward

    scores_deque.append(rewards)

    '''accumulates batches of scores'''

    if i % print_every == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_deque)))
    if np.mean(scores_deque) >= 195.0:
        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i - 100,
                                                                                   np.mean(scores_deque)))
        break
