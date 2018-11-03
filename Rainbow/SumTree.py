import numpy as np
import torch
from collections import namedtuple
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

'''

https://stackoverflow.com/questions/8947153/efficient-algorithm-for-random-sampling-from-a-distribution-while-allowing-updat
'''
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.beta = 0.4
        self.beta_incrementor = 0.001

    def get_max_p(self):
        '''
        get max priority value. This might be an inefficient way. Max Heap might be better?
        :param tree:
        :return:
        '''
        if(self.write > 0):
            max_priority = max(self.tree[self.capacity -1: self.capacity + self.write -1])
        else:
            max_priority = 1
        return max_priority

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def sample(self, n):
        batch = []
        idxs = []

        segment = self.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_incrementor])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)

            while True:
                '''
                exception handling 
                '''
                try:
                    (idx, p, data) = self.get(s)
                except TypeError:
                    print("Sampled a weird data: ", data)
                else:
                    break

            priorities.append(p)

            if(i == 0):
                state = data[0]
                action = data[1]
                reward = data[2]
                next_state = data[3]
                done = data[4]

            try:
                state = np.vstack((state, data[0]))
                action = np.vstack((action, data[1]))
                reward = np.vstack((reward, data[2]))
                next_state = np.vstack((next_state, data[3]))
                done = np.vstack((done, data[4]))
            except TypeError:
                print(data)


            idxs.append(idx)

        state = torch.from_numpy(state).float().to(device)
        action = torch.from_numpy(action).long().to(device)
        reward = torch.from_numpy(reward).float().to(device)
        next_state = torch.from_numpy(next_state).float().to(device)
        done = torch.from_numpy(done.astype(np.uint8)).float().to(device)



        sampling_probabilities = priorities / self.total()
        is_weight = np.power(self.write * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        is_weight = torch.from_numpy(is_weight).float().to(device)

        return state, action, reward, next_state, done, idxs, is_weight
