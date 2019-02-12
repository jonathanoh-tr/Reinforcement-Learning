
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.distributions import Categorical

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out, Activation=True, Norm=False):
        super(Linear, self).__init__()

        steps = [nn.Linear(dim_in, dim_out)]

        if Norm != False:
            steps.append(nn.BatchNorm1d(dim_out))

        if Activation != False:
            steps.append(nn.ReLU())

        self.model = nn.Sequential(*steps)

    def forward(self, x):

        return self.model(x)

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()

        steps = [Linear(s_size, h_size[0])]

        if len(h_size) > 1:
            for i in range(len(h_size)-1):
                steps.append(Linear(h_size[i], h_size[i+1]))

        steps.append(Linear(h_size[-1], a_size, Activation=False))

        self.model = nn.Sequential(*steps)

    def forward(self, x):

        return F.softmax(self.model(x), dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()

        return action.item(), m.log_prob(action)

    def get_prob(self, state, action):

        state = state
        probs = self.forward(state)
        m = Categorical(probs)

        return m.log_prob(action.squeeze())
