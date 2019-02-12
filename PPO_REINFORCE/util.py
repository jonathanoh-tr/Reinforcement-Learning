
import numpy as np
import torch
import torch.optim.optimizer

def render_text_envq(env, agent):
    env.seed(np.random.randint(0, 10000))
    state = env.reset()
    rewards =[]
    while True:
        env.render()
        max_action = agent.act(state)
        state, reward, done, info = env.step(max_action[0])
        rewards.append(reward)
        if (done):
            print("Environment Terminated")
            print("Total Rewards: ", np.sum(rewards))
            break

def cumulative_discounted_rewards_togo(rewards, gamma):
    '''
    caculates and returns an array with len == len(trajectory) where each element is cumulated discounted rewards to go.
    Only works for 1D rewards array
    :param rewards: An array or a list of rewards
    :param gamma: discount rate.
    :return:
    '''
    discount = []
    cumulative_sum = []

    for i in range(0, len(rewards)):
        discount.append(gamma ** i)

    for i in range(1, len(rewards) + 1):
        cumulative_sum.append(np.matmul(rewards[::-1][0:i], discount[0:i][::-1]))

    return cumulative_sum[::-1]

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

def PPO_Update(policy, memory, optimizer, opts):

    for i in range(opts.epoch):
        state, action, reward, probs, done = memory.sample()

        new_probs = policy.get_prob(state, action)

        ratios = (new_probs - probs.squeeze()).exp()

        obs1 = ratios * reward
        obs2 = torch.clamp(ratios, 1 - opts.eps, 1 + opts.eps) * reward

        obs = - torch.min(obs1, obs2).mean()

        optimizer.zero_grad()
        obs.backward()
        optimizer.step()