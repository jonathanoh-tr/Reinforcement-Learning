
import gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agent import Agent

env = gym.make('CartPole-v1')

agent = Agent(env.observation_space.shape[0], env.action_space.n, seed=0, batch_size=32, q_size=25)

env.seed(0)

def DQN(num_episodes = 7500, max_iteration = 1000, init_epsilon = 1.0, min_epsilon = 0.05, decay = 0.999):
    '''

    :param num_episodes:
    :param max_iteration:
    :param init_epsilon:
    :param min_epsilon:
    :param decay:
    :return:
    '''


    total_reward = []
    total_reward_window = deque(maxlen=100)
    epsilon = init_epsilon

    for i in range(num_episodes):
        state = env.reset()
        rewards = 0

        for k in range(max_iteration):

            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)

            agent.step(state, action, reward, next_state, done)

            state = next_state
            rewards += reward

            if done:
                break

        total_reward_window.append(rewards)
        total_reward.append(rewards)

        epsilon = max(min_epsilon, epsilon * decay)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(total_reward_window)), end="")
        if i % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i, np.mean(total_reward_window)))

        if np.mean(total_reward_window) >= 195:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i - 100,
                                                                                         np.mean(total_reward_window)))
            torch.save(agent.current_model.state_dict(), 'checkpoint.pth')
            break
    torch.save(agent.current_model.state_dict(), 'checkpoint_end.pth')
    return total_reward


scores = DQN()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


def render_text_envq(env):
    env.seed(12456)
    state = env.reset()
    rewards = 0
    while True:
        env.render()

        max_action = agent.act(state, -1)
        state, reward, done, info = env.step(max_action)
        rewards += reward
        if (done):
            print("Environment Terminated")
            print("Total Rewards: ", rewards)
            break
    env.render()


render_text_envq(env)
env.close()
