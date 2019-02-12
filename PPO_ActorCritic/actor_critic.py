import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class actor_critic(nn.Module):

    def __init__(self, state_space, action_space, seed, first_layer=16, second_layer=16):

        super(actor_critic, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.seed = torch.manual_seed(seed)

        '''separate nn version'''
        self.policy = nn.Sequential(
            nn.Linear(state_space, first_layer),
            nn.ReLU(),
            nn.Linear(first_layer, action_space)
        )

        self.value = nn.Sequential(
            nn.Linear(state_space, first_layer),
            nn.ReLU(),
            nn.Linear(first_layer, second_layer),
            nn.BatchNorm1d(second_layer),
            nn.ReLU(),
            nn.Linear(second_layer, 1),
        )

    def forward(self, state, output_choice):
        '''
        Given a state vector and output_choice, does a forward pass on a NN that's chosen
        :param state:
        :param output_choice:
        :return:
        '''

        if (output_choice == "policy"):
            '''policy'''
            output = F.softmax(self.policy(state), dim=1)
        else:
            '''value'''
            output = self.value(state)

        return output

    def act(self, state):
        '''
        Takes in state, and outputs action (stochastic), and log_prob of that action
        :param state:
        :return:
        '''

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        actions = self.forward(state, output_choice='policy').cpu()
        choices = Categorical(actions)

        action = choices.sample()

        return action.item(), choices.log_prob(action)

    def get_prob(self, state, action):

        state = state
        probs = self.forward(state, output_choice='policy')
        m = Categorical(probs)

        return m.log_prob(action.squeeze())