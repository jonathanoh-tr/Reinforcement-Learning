import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

#USE_CUDA = torch.cuda.is_available()
USE_CUDA = not torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class NoisyLinear(nn.Module):
    '''
    A noisy net with factored Gaussian noise as described in the paper
    '''

    def __init__(self, input_features, output_features, std_init=0.4):

        super(NoisyLinear, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(output_features, input_features).to(device))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_features, input_features).to(device))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_features, input_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(output_features).to(device))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_features).to(device))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_features).to(device))

        self.reset_parameters()
        self.reset_noise()


    def forward(self, x):

        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        '''3.2 Initialization of Noisy Networks'''
        '''Each element was initialized by a sample from an independent uniform distribution '''
        '''U[-1/sqrt(p), 1/sqrt(p)] where p is the number of elements'''
        '''each element sigma was initialised to a constant sigma / sqrt(p)'''
        '''sigma set to 0.5'''

        '''initialize mu with uniform(-1/sqrt(p), 1/sqrt(p))'''
        mu_range = 1/ math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)

        '''sigma is given by the initializer, std_init'''
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        '''initilize mu with uniform'''
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def _scale_noise(self, size):
        '''factorized Gaussian Noise'''
        '''f(x) = sgn(x) * sqrt(|x|)'''

        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())

        return x

    def reset_noise(self):
        '''factorized Gaussian Noise'''

        epsilon_in = self._scale_noise(self.input_features)
        epsilon_out = self._scale_noise(self.output_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        '''.ger is the outer product'''
        '''takes in a noise vector of size p and a noise vector of size q'''
        '''returns the outer product which is p X q matrix of noise'''

        self.bias_epsilon.copy_(self._scale_noise(self.output_features))



