import gym
import torch
import numpy as np
import torch.multiprocessing as mp

from actor_critic import actor_critic
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#TODO Make the model and training more modular

def online_A3C(shared_model, shared_optimizer,  env, seed, n_episodes, max_t, gamma, print_every, pos):

    from collections import deque

    optimizer = shared_optimizer

    shared_model.cuda()
    env.seed(seed)
    scores_deque = deque(maxlen=100)
    scores = []
    saved_log_probs = []

    for i_episode in range(1, n_episodes + 1):
        rewards = []
        state = env.reset()

        for t in range(max_t):
            action, log_prob = shared_model.act(state)
            saved_log_probs.append(log_prob)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            '''Reward to go'''

            rewards_batch = torch.from_numpy(np.asarray(reward)).float().to(device)
            states = torch.from_numpy(np.asarray(state)).float().to(device)
            next_states = torch.from_numpy(np.asarray(next_state)).float().to(device)

            val_estimate = shared_model.forward(states, output_choice='val')
            val_estimate_next = shared_model.forward(next_states, output_choice='val')

            loss = F.mse_loss(val_estimate, rewards_batch + gamma * val_estimate_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_estimate = shared_model.forward(states, output_choice='val')
            val_estimate_next = shared_model.forward(next_states, output_choice='val')

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
            print(pos, seed)
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            break

    return scores



if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    #set the seed for the model
    torch.manual_seed(1)

    '''
    .cpu() is to avoid getting an error message
    RuntimeError: Cannot pickle CUDA storage; try pickling a CUDA tensor instead
    '''
    model = actor_critic(env.observation_space.shape[0], env.action_space.n).cpu()
    #have the model useable by all processes
    model.share_memory()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    #number of processes
    num_processes = 5
    processes = []

    seed_list = np.arange(0, 5, dtype=int)
    for rank, seed in zip(range(num_processes), seed_list):
        p = mp.Process(target=online_A3C, args=(model, optimizer, env, int(seed), 100, 500, 1, 5, rank))
        p.start()
        processes.append(p)





