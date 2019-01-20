import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--batch', type=int, nargs='?', default=128,
                                 help='batch size to be used')

        self.parser.add_argument('--num_episodes', type=int, nargs='?', default=5000)
        self.parser.add_argument('--max_steps', type=int, nargs='?', default=1000)
        self.parser.add_argument('--frame_idx', type=int, nargs='?', default=0)
        self.parser.add_argument('--print_every', type=int, nargs='?', default=5)
        self.parser.add_argument('--threshold', type=float, nargs='?', default=0.0)

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt