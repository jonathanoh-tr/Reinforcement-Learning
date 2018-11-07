
import gym
import torch
from util import NormalizedActions
from agent import Agent

env = NormalizedActions(gym.make('Pendulum-v0'))

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

num_episodes  = 5000
max_steps   = 500
frame_idx   = 0
rewards     = []
batch_size  = 128

for eps in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):

        if eps % 100 == 0:
            action = policy.act(state, step, False)
        else:
            action = policy.act(state, step)
        next_state, reward, done, _ = env.step(action)

        policy.add_to_memory(state, action, reward, next_state, done)
        if policy.memory.__len__() > batch_size:
            policy.update(step)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        if frame_idx % 1000 == 0:
            plot(eps, rewards)

        if done:
            if eps % 100 == 0:
                print("\nThis Episode :", eps)
                print("\nRewards without noise is: ", episode_reward)
            break

    rewards.append(episode_reward)