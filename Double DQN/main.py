
import gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from agent import Agent
import util

from options import options

options = options()
opts = options.parse()

env = gym.make(opts.env)

agent = Agent(env.observation_space.shape[0], env.action_space.n, opts=opts, seed=0)

env.seed(opts.env_seed)

def DDQN(num_episodes = opts.num_episodes, max_iteration = opts.max_iteration, init_epsilon = 1.0, min_epsilon = opts.min_epsilon, decay = opts.decay):
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

        if np.mean(total_reward_window) >= opts.win_cond:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i - 100,
                                                                                         np.mean(total_reward_window)))
            torch.save(agent.local_model.state_dict(), 'checkpoint.pth')
            break

    torch.save(agent.local_model.state_dict(), 'checkpoint_end.pth')
    return total_reward


scores = DDQN()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

if opts.render == True:
    util.render_text_envq(env, agent)
    env.close()