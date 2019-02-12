
import numpy as np
import torch
import torch.optim.optimizer
import torch.nn.functional as F

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

        state, action, reward, next_state, probs, done = memory.sample()

        val_estimate = policy.forward(state, output_choice='val')
        val_estimate_next = policy.forward(next_state, output_choice='val')

        loss = F.mse_loss(val_estimate, reward.view(reward.size(0), -1) + opts.discount_rate * val_estimate_next)

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()

        new_probs = policy.get_prob(state, action)
        ratios = (new_probs - probs.squeeze()).exp()

        advantage = reward + opts.discount_rate * val_estimate_next.detach() - val_estimate.detach()
        obs1 = ratios * advantage
        obs2 = torch.clamp(ratios, 1 - opts.eps, 1 + opts.eps) * advantage

        obs = - torch.min(obs1, obs2).mean()

        optimizer[1].zero_grad()
        obs.backward()
        optimizer[1].step()