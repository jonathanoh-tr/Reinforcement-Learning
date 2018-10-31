
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


'''initialize the layer by + - 1/sqrt(fan_in)'''

def hidden_initilizer(layer):

    fan_in = layer.weight.data.size()[0]
    lim = 1./ np.sqrt(fan_in)

    return (-lim, lim)

class Actor(nn.Module):
    '''Actor (Policy) model'''

    def __init__(self, state_space, action_space, seed, hidden_layer=256):

        super(Actor, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.seed = torch.manual_seed(seed)

        self.FC_layer1 = nn.Linear(state_space, hidden_layer)
        self.FC_layer2 = nn.Linear(hidden_layer, action_space)
        self.initialize_parameters()

    def initialize_parameters(self):

        '''section 7 in the paper
        The final layer weights and biases of both the actor and critic
        were initialized from a uniform distribution'''

        self.FC_layer1.weight.data.uniform_(hidden_initilizer(self.FC_layer1))
        self.FC_layer2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):

        output = nn.ReLU(self.FC_layer1(state))

        return F.tanh(self.FC_layer2(output))

class Critic(nn.Module):

    def __init__(self, state_space, action_space, seed, hidden_layer1=256, hidden_layer2=256, hidden_layer3=128):

        super(Critic, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.seed = torch.manual_seed(seed)
        self.FC_layer1 = nn.Linear(state_space, hidden_layer1)
        self.FC_layer2 = nn.Linear(hidden_layer1 + action_space, hidden_layer2)
        self.FC_layer3 = nn.Linear(hidden_layer2, hidden_layer3)
        self.FC_layer4 = nn.Linear(hidden_layer3, 1)

    def initialize_parameters(self):

        self.FC_layer1.weight.data.uniform_(hidden_initilizer(self.FC_layer1))
        self.FC_layer2.weight.data.uniform_(hidden_initilizer(self.FC_layer2))
        self.FC_layer3.weight.data.uniform_(hidden_initilizer(self.FC_layer3))
        self.FC_layer4.weight.data.uniform_((-3e-3, 3e-3))

    def forward(self, state, action):

        output = nn.ReLU(self.FC_layer1(state))
        output = nn.ReLU(self.FC_layer2(output + action))
        output = nn.ReLU(self.FC_layer3(output))

        return self.FC_layer4(output)





