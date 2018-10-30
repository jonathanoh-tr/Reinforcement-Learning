
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QDQN(nn.Module):

    def __init__(self, state_space, action_space, hidden_layer_1=128, hidden_layer_2=128, n_quantiles=51):

        super(QDQN, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.n_quantiles = n_quantiles

        self.linear = nn.Linear(state_space, hidden_layer_1)

        self.linear1 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.linear2 = nn.Linear(hidden_layer_2, action_space * n_quantiles)

    def forward(self, state):

        output = F.relu(self.linear(state))
        output = F.relu(self.linear1(output))
        output = self.linear2(output)

        return output.view(-1, self.action_space, self.n_quantiles)

    def act(self, state):

        state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
        '''volatile used when you know you won't be calling backprop'''
        q_value = self.forward(state).mean(2)
        '''average over quantiles and then pick an action'''
        action = q_value.max(1)[1].data[0]
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()




