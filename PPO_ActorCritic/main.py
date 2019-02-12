import gym
from collections import deque

#import the policy model
from actor_critic import actor_critic
#from model import Policy

#import options
from options import options
options = options()
opts = options.parse()

#import util module
from util import *

import torch
torch.manual_seed(0) # set random seed
import torch.optim as optim

env = gym.make(opts.env)
env.seed(opts.env_seed)

#import memory for ppo
from memory import replayMemory
memory = replayMemory(env.action_space.n, memory_size = opts.max_iteration, batch_size=opts.batch, seed=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_size=env.observation_space.shape[0]
policy = actor_critic(state_space=state_size, action_space=env.action_space.n, seed=0).to(device)

if opts.print_model:
    print("The model: ", policy)

optimizer_policy = optim.Adam(policy.policy.parameters(), lr=opts.lr)
optimizer_value = optim.Adam(policy.value.parameters(), lr=opts.lr)

optimizer = [optimizer_value, optimizer_policy]

def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []

    for i_episode in range(1, n_episodes + 1):
        saved_log_probs = []

        states = []
        actions = []
        rewards = []
        next_states = []
        log_probs = []
        dones = []
        steps = 0

        memory.reset()
        state = env.reset()

        for t in range(max_t):

            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            log_probs.append(log_prob.detach().cpu().numpy())
            dones.append(done)

            state = next_state

            steps += 1

            if done:
                break

            if steps == opts.max_step:
                R = cumulative_discounted_rewards_togo(rewards, gamma=gamma)

                for i in range(len(states)):
                    memory.add(states[i], actions[i], R[i], next_states[i], log_probs[i], dones[i])

                PPO_Update(policy, memory, optimizer, opts)

            memory.reset()

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        R = cumulative_discounted_rewards_togo(rewards, gamma=gamma)

        for i in range(len(states)):
            memory.add(states[i], actions[i], R[i], next_states[i], log_probs[i], dones[i])

        PPO_Update(policy, memory, optimizer, opts)

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= opts.win_cond:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores

scores = reinforce(n_episodes=opts.num_episodes, max_t=opts.max_iteration, gamma=opts.discount_rate, print_every=opts.print_every)

if opts.render:
    render_text_envq(env, policy)
