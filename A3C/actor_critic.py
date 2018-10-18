
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class actor_critic(nn.Module):

    def __init__(self, state_space, action_space, first_layer=16, second_layer=16):

        super(actor_critic, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        '''separate nn version'''
        self.pol_nn1 = nn.Linear(state_space, first_layer).to(device)
        self.pol_nn2 = nn.Linear(first_layer, action_space).to(device)

        self.val_nn1 = nn.Linear(state_space, first_layer).to(device)
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
            output = F.relu(self.pol_nn1(state))
            output = F.softmax(self.pol_nn2(output), dim=1)
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

        actions = self.forward(state, output_choice='policy').cpu()
        choices = Categorical(actions)

        action = choices.sample()

        return action.item(), choices.log_prob(action)

