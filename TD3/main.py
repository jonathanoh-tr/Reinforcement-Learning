
import gym
import torch
from util import NormalizedActions
from collections import deque
from agent import Agent
import numpy as np

from options import options


options = options()

opts = options.parse()
batch = opts.batch

env = NormalizedActions(gym.make('BipedalWalker-v2'))

from IPython.display import clear_output
import matplotlib.pyplot as plt

policy = Agent(env)
def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('Episode %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

rewards = []

for eps in range(opts.num_episodes):
    scores_deque = deque(maxlen=100)
    scores = []
    state = env.reset()
    episode_reward = 0

    for step in range(opts.max_steps):

        if eps % 100 == 0:
            action = policy.act(state, step, False)
        else:
            action = policy.act(state, step)

        next_state, reward, done, _ = env.step(action)
        policy.add_to_memory(state, action, reward, next_state, done)

        if policy.memory.__len__() > opts.batch:
            policy.update(step)

        state = next_state
        episode_reward += reward
        opts.frame_idx += 1

        if done:
            break

    scores_deque.append(episode_reward)
    scores.append(episode_reward)

    if eps % opts.print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(eps, np.mean(scores_deque)))

    if np.mean(scores_deque) >= opts.threshold:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(eps - 100,
                                                                                     np.mean(scores_deque)))

        break

    rewards.append(episode_reward)