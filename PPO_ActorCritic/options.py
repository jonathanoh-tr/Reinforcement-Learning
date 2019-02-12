import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        #Agent Options
        self.parser.add_argument('--batch', type=int, nargs='?', default=5, help='batch size to be used')
        self.parser.add_argument('--memory_size', type=int, nargs='?', default=1000000, help='size of replay memory')
        self.parser.add_argument('--update_freq', type=int, nargs='?', default=4, help='how often to update the model')
        self.parser.add_argument('--lr', type=float, nargs='?', default=0.01, help='learning rate')
        self.parser.add_argument('--discount_rate', type=float, nargs='?', default=0.90, help='rewards discount rate')
        self.parser.add_argument('--transfer_rate', type=float, nargs='?', default=0.001, help='transfer rate for soft update')
        self.parser.add_argument('--hidden', type=int, nargs='+', default=[16], help='hidden layer configuration in a list form')
        self.parser.add_argument('--epoch', type=int, nargs='?', default=5, help='number of ppo updates')
        self.parser.add_argument('--eps', type=float, nargs='?', default=0.20, help='epsilon for PPO')

        #Env Options
        self.parser.add_argument('--env', type=str, nargs='?', default='CartPole-v1', help='Name of the OpenAI Env')
        self.parser.add_argument('--env_seed', type=int, nargs='?', default=0, help='random seed for the environment')

        #Training Options
        self.parser.add_argument('--num_episodes', type=int, nargs='?', default=3000, help='total number of training episodes')
        self.parser.add_argument('--max_step', type=int, nargs='?', default=20, help='max number of iterations per episodes')
        self.parser.add_argument('--max_iteration', type=int, nargs='?', default=1000, help='max number of iterations per episodes')
        self.parser.add_argument('--min_epsilon', type=float, nargs='?', default=0.1, help='min value for epsilon')
        self.parser.add_argument('--decay', type=float, nargs='?', default=0.995, help='decay rate of epsilon per episode')
        self.parser.add_argument('--win_cond', type=int, nargs='?', default=200, help='Condition where the env is considered solved')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=100, help='How often to print scores')
        self.parser.add_argument('--print_model', type=bool, nargs='?', default=True, help='Prints the model being used')
        #render
        self.parser.add_argument('--render', type=bool, nargs='?', default=True, help='Renders an episode at the end of training')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt



"""
options = options()
opts = options.parse()
batch = opts.batch
"""