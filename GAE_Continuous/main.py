
import gym
import torch
from collections import deque
import numpy as np

from pytorch_actor_critic import actor_critic
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make("Pendulum-v0")
env.seed(0)

policy = actor_critic(env, seed=86)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)


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



def compute_gae(next_value, rewards, done, values, gamma=0.99, tau=0.95):

    values = torch.cat((values, next_value.unsqueeze(1)))
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * done[step] - values[step]
        gae = delta + gamma * tau * done[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def reinforce_loop(n_episodes=1500, max_t=1000, gamma=1, print_every=5, batch_size=1):

    scores_deque = deque(maxlen=100)
    scores = []

    saved_log_probs = []
    reward_array = []


    for i_episode in range(n_episodes):
        states = []
        rewards = []
        next_states = []
        dones = []
        entropys = 0
        state = env.reset()
        states.append(state)

        for t in range(max_t):
            action, log_prob, entropy = policy.act(state)
            action = policy.un_normalizer(action, env.action_space.low[0], env.action_space.high[0])
            entropys += entropy
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            dones.append(done)
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

            '''GAE'''

            rewards_tensor = torch.from_numpy(np.asarray(rewards)).float().to(device)
            state_tensor = torch.from_numpy(np.asarray(states)).float().to(device)
            next_state_tensor = torch.from_numpy(np.asarray(next_states)).float().to(device)
            dones_tensor = torch.from_numpy(np.asarray(dones).astype(np.uint8)).float().to(device)

            val_estimate = policy.forward(state_tensor, output_choice='val')
            val_estimate_next = policy.forward(next_state_tensor[-1], output_choice='val')

            rewards_tensor = rewards_tensor.unsqueeze(1)
            dones_tensor = dones_tensor.unsqueeze(1)

            policy_loss = []

            gae_returns = compute_gae(val_estimate_next, rewards_tensor, dones_tensor, val_estimate)
            gae_returns = torch.stack(gae_returns)

            for idx, log_prob in enumerate(saved_log_probs):
                policy_loss.append(-log_prob.to(device) * gae_returns)
                #policy_loss.append(-log_prob.to(device) * (gae_returns[idx] + gamma * val_estimate_next[idx] - val_estimate[idx]))


            '''Train Policy'''
            policy_loss = torch.cat(policy_loss).mean()

            '''Train Value'''
            value_loss = F.mse_loss(val_estimate, rewards_tensor.view(rewards_tensor.size(0), -1) + gamma * val_estimate_next)

            loss = policy_loss + 0.5 * value_loss - 0.001 * entropys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            saved_log_probs = []
            reward_array = []

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 500.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores

scores = reinforce_loop()




def render_text_envq(env):
    env.seed(12456)
    state = env.reset()
    rewards = 0
    while True:
        env.render()

        action, log_prob, entropy = policy.act(state)
        action = policy.un_normalizer(action, env.action_space.low[0], env.action_space.high[0])
        state, reward, done, info = env.step(action)
        rewards += reward
        if (done):
            print("Environment Terminated")
            print("Total Rewards: ", rewards)
            break
    env.render()


render_text_envq(env)
env.close()


