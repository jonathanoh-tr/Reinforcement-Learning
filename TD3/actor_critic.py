
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_layer=256, hidden_layer1=256, w_init=3e-3):
        super(Actor, self).__init__()
        self.linear  = nn.Linear(state_size, hidden_layer).to(device)
        self.linear1 = nn.Linear(hidden_layer, hidden_layer1).to(device)
        self.linear2 = nn.Linear(hidden_layer1, action_size).to(device)

        self.linear2.weight.data.uniform_(-w_init, w_init)
        self.linear2.bias.data.uniform_(-w_init, w_init)

    def forward(self, state):

        action = F.relu(self.linear(state))
        action = F.relu(self.linear1(action))
        action = torch.tanh(self.linear2(action))

        return action

    def get_action(self, state):
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


class Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden_layer=256, hidden_layer1=256, w_init=3e-3):
        super(Critic, self).__init__()

        self.linear     = nn.Linear(state_size + action_size, hidden_layer).to(device)
        self.linear1    = nn.Linear(hidden_layer, hidden_layer1).to(device)
        self.linear2    = nn.Linear(hidden_layer1, 1).to(device)

        self.linear2.weight.data.uniform_(-w_init, w_init)
        self.linear2.bias.data.uniform_(-w_init, w_init)

    def forward(self, state, action):

        value   = torch.cat([state, action], 1)
        value   = F.relu(self.linear(value))
        value   = F.relu(self.linear1(value))
        value   = self.linear2(value)

        return value





