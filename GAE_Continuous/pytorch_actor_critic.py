
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-3
class actor_critic(nn.Module):

    def __init__(self, env, seed, first_layer=256, second_layer=256):

        super(actor_critic, self).__init__()

        self.env = env
        self.state_space = env.env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.seed = torch.manual_seed(seed)

        '''separate nn version'''
        self.pol_nn1 = nn.Linear(self.state_space, first_layer).to(device)
        self.pol_nn2 = nn.Linear(first_layer, self.action_space).to(device)


        self.val_nn1 = nn.Linear(self.state_space, first_layer).to(device)
        self.val_nn2 = nn.Linear(first_layer, second_layer).to(device)
        self.val_nn3 = nn.Linear(second_layer, 1).to(device)

    def forward(self, state, output_choice):
        '''
        Given a state vector and output_choice, does a forward pass on a NN that's chosen

        :param state:
        :param output_choice:
        :return:
        '''

        if (output_choice == "policy"):
            '''policy'''
            std_activation = nn.Softplus()

            output = F.relu(self.pol_nn1(state))
            mu = self.pol_nn2(output)
            std = std_activation(mu) + EPSILON
            output = torch.distributions.Normal(mu, std)

        else:
            '''value'''

            output = F.relu(self.val_nn1(state))
            output = F.relu(self.val_nn2(output))
            output = self.val_nn3(output)

        return output

    def act(self, state):
        '''
        Takes in state, and outputs action (stochastic), and log_prob of that action
        :param state:
        :return:
        '''

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)


        actions = self.forward(state, output_choice='policy')

        action = actions.sample()
        log_prob = actions.log_prob(action)
        entropy = actions.entropy()

        return action.cpu(), log_prob, entropy

    def normalize(self, action, low, high):
        '''
        Normalizes the inputs between [low, high] to be between [-1, 1]
        :param action:
        :param low:
        :param high:
        :return:
        '''

        action = 2 * (action - low) / (high - low) -1

        return action

    def un_normalizer(self, action, low, high):

        action = (action + 1.0) * 0.5 * (high - low) + low
        action = np.clip(action, low, high)

        return action

