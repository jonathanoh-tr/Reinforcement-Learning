
import gym
import torch
from collections import deque
import numpy as np

from actor_critic import actor_critic
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1')
env.seed(5)

policy = actor_critic(env.observation_space.shape[0], env.action_space.n, seed=5)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

def reinforce_loop(n_episodes=1500, max_t=1000, gamma=1, print_every=100):

    scores_deque = deque(maxlen=100)
    scores = []
    saved_log_probs = []

    for i_episode in range(1, n_episodes + 1):
        rewards = []
        state = env.reset()

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            '''Reward to go'''

            rewards_batch = torch.from_numpy(np.asarray(reward)).float().to(device)
            states = torch.from_numpy(np.asarray(state)).float().to(device)
            next_states = torch.from_numpy(np.asarray(next_state)).float().to(device)

            val_estimate = policy.forward(states, output_choice='val')
            val_estimate_next = policy.forward(next_states, output_choice='val')

            loss = F.mse_loss(val_estimate, rewards_batch + gamma * val_estimate_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_estimate = policy.forward(states, output_choice='val')
            val_estimate_next = policy.forward(next_states, output_choice='val')

            policy_loss = []

            policy_loss.append(-log_prob.to(device) * (rewards_batch + gamma * val_estimate_next - val_estimate))

            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            saved_log_probs = []

            state = next_state

            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        '''accumulates batches of scores'''

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores

scores = reinforce_loop()



