
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPSILON = 1e-6
STD_MIN = -20
STD_MAX = 2

class Value(nn.Module):
    
    def __init__(self, state_size, hidden_layer=256, hidden_layer1=256, w_init=3e-3):
        super(Value, self).__init__()
        self.linear     = nn.Linear(state_size, hidden_layer).to(device)
        self.linear1    = nn.Linear(hidden_layer, hidden_layer1).to(device)
        self.linear2    = nn.Linear(hidden_layer1, 1).to(device)

        '''Same init weight as DDPG'''
        self.linear2.weight.data.uniform_(-w_init, w_init)
        self.linear2.bias.data.uniform_(-w_init, w_init)

    def forward(self, state):

        output = F.relu(self.linear(state))
        output = F.relu(self.linear1(output))
        output = self.linear2(output)

        return output

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_layer=256, hidden_layer1=256, w_init=3e-3):
        super(Policy, self).__init__()

        self.linear  = nn.Linear(state_size, hidden_layer).to(device)
        self.linear1 = nn.Linear(hidden_layer, hidden_layer1).to(device)


        self.mean_output = nn.Linear(hidden_layer1, action_size).to(device)
        '''Same init weight as DDPG'''
        self.mean_output.weight.data.uniform_(-w_init, w_init)
        self.mean_output.bias.data.uniform_(-w_init, w_init)

        self.std_output = nn.Linear(hidden_layer1, action_size).to(device)
        self.std_output.weight.data.uniform_(-w_init, w_init)
        self.std_output.bias.data.uniform_(-w_init, w_init)

    def forward(self, state):

        output = F.relu(self.linear(state))
        output = F.relu(self.linear1(output))

        mean    = self.mean_output(output)
        log_std = self.std_output(output)
        log_std = torch.clamp(log_std, STD_MIN, STD_MAX)

        return mean, log_std

    def evaluate(self, state):

        mean, log_std = self.forward(state)
        std = log_std.exp()

        '''reparameterization'''
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def act(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal  = Normal(mean, std)
        z       = normal.sample()
        action  = torch.tanh(z)

        action = action.detach().cpu().numpy()

        return action[0]

class soft_Q(nn.Module):    
    def __init__(self, state_size, action_size, hidden_layer=256, hidden_layer1=256, w_init=3e-3):
        super(soft_Q, self).__init__()

        self.linear  = nn.Linear(state_size + action_size, hidden_layer).to(device)
        '''input is state_size + action_size'''
        self.linear1 = nn.Linear(hidden_layer, hidden_layer1).to(device)
        self.linear2 = nn.Linear(hidden_layer1, 1).to(device)

        self.linear2.weight.data.uniform_(-w_init, w_init)
        self.linear2.bias.data.uniform_(-w_init, w_init)

    def forward(self, state, action):

        output = torch.cat([state, action], 1)
        '''torch.concat to concatenate state and action tensors'''
        output = F.relu(self.linear(output))
        output = F.relu(self.linear1(output))
        output = self.linear2(output)

        return output
