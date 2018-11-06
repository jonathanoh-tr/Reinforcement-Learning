import gym
import numpy as np

class Action_Normalizer(gym.ActionWrapper):
    '''
    Class used to normalize actions of an env to make the range fall between [-1, 1]
    '''

    def _reverse_action(self, action):
        low     = self.action_space.low
        high    = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action

    def _action(self, action):
        low     = self.action_space.low
        high    = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action