
import gym
import torch
from collections import deque
import numpy as np

from pytorch_actor_critic_shared import actor_critic
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#TODO Causality #1) reward to go. Use cumulative rewards from current to the end.
'''cumulative_discounted_rewards_togo solves the issue above'''
#TODO Causality #2) need an efficient way to calculate the cumulative sum of rewards to get the rewards to go.
'''need to see how to improve cumulative_discounted_rewards_togo() to be more efficient '''

def cumulative_discounted_rewards_togo(array, gamma):
    '''

    caculates and returns an array with len == len(trajectory) where each element is cumulated discounted rewards to go.

    :param array:
    :param gamma:
    :return:
    '''
    discount = []
    cumulative_sum = []

    for i in range(0, len(array)):
        discount.append(gamma ** i)

    for i in range(1, len(array) + 1):
        cumulative_sum.append(np.matmul(array[::-1][0:i], discount[0:i][::-1]))

    return cumulative_sum[::-1]

#TODO Batch #3) current accumulates 1 trajectory and update the model, update it so that it accumulates n number of trajectories
#TODO Baseline #4) Subtract average baseline after getting n number of batches of trajectories.

def batch_mean_adjusting(reward_array, discount, log_probs):
    '''
    :param scores:
    :param reward_array:
    :param discount:
    :param log_probs:
    :param batch_size:
    :return:
    '''

    batch_size = len(reward_array)

    total_reward = []
    for i in range(batch_size):
        total_reward.append(sum(reward_array[i]))

    average_reward = np.mean(total_reward)


    adjusted_rewards = []
    cumulative_rewards = []
    for i in range(batch_size):
        mean_adjusted = (np.asarray(total_reward[i]) - average_reward)
        adjusted_rewards.append(np.asarray(reward_array[i]) * (mean_adjusted / np.asarray(total_reward[i])))
        cumulative_rewards.append(cumulative_discounted_rewards_togo(adjusted_rewards[i], discount))
    batch_rewards = np.concatenate(cumulative_rewards, axis=0)

    policy_loss = []
    for idx, log_prob in enumerate(log_probs):
        policy_loss.append(-log_prob * batch_rewards[idx])

    return policy_loss, batch_rewards


env = gym.make('CartPole-v1')
env.seed(5)

policy = actor_critic(env.observation_space.shape[0], env.action_space.n, seed=5)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def reinforce_loop(n_episodes=1500, max_t=1000, gamma=1, print_every=100, batch_size=2):

    scores_deque = deque(maxlen=100)
    scores = []
    reward_array = []
    saved_log_probs = []
    states = []
    next_states = []

    for i_episode in range(1, n_episodes + 1):
        rewards = []
        state = env.reset()
        states.append(state)

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            next_states.append(state)

            if not done:
                states.append(state)

            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        reward_array.append(rewards)
        '''accumulates batches of scores'''

        if (i_episode >= batch_size and i_episode % batch_size == 0):

            '''Reward to go'''
            rewards_batch = []
            for i in range(batch_size):
                rewards_batch.append(cumulative_discounted_rewards_togo(reward_array[i], gamma=gamma))
            rewards_batch = np.concatenate(rewards_batch, axis = 0)
            rewards_batch = torch.from_numpy(np.asarray(rewards_batch)).float().to(device)

            states = torch.from_numpy(np.asarray(states)).float().to(device)
            next_states = torch.from_numpy(np.asarray(next_states)).float().to(device)

            val_estimate = policy.forward(states, output_choice='val')
            val_estimate_next = policy.forward(next_states, output_choice='val')

            loss = F.mse_loss(val_estimate, rewards_batch.view(rewards_batch.size(0), -1) + gamma * val_estimate_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_estimate = policy.forward(states, output_choice='val')
            val_estimate_next = policy.forward(next_states, output_choice='val')

            policy_loss = []

            for idx, log_prob in enumerate(saved_log_probs):
                policy_loss.append(-log_prob.to(device) * (rewards_batch[idx] + gamma * val_estimate_next[idx] - val_estimate[idx]))

            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            saved_log_probs = []
            reward_array = []
            states = []
            next_states = []

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores

scores = reinforce_loop()



