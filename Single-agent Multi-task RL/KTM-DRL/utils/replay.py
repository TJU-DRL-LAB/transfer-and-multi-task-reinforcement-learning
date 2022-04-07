import numpy as np
import torch

# https://github.com/sfujim/TD3/blob/ade6260da88864d1ab0ed592588e090d3d97d679/utils.py

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.state[ind]).float().to(self.device),
            torch.from_numpy(self.action[ind]).float().to(self.device),
            torch.from_numpy(self.next_state[ind]).float().to(self.device),
            torch.from_numpy(self.reward[ind]).float().to(self.device),
            torch.from_numpy(self.not_done[ind]).float().to(self.device)
        )

    def sample_np(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            np.float32(self.state[ind]),
            np.float32(self.action[ind]),
            np.float32(self.next_state[ind]),
            np.float32(self.reward[ind]),
            np.float32(self.not_done[ind])
        )

    def save(self, fdir):
        np.save(fdir + '/sample-state', self.state[:self.size])
        np.save(fdir + '/sample-action', self.action[:self.size])
        np.save(fdir + '/sample-nstate', self.next_state[:self.size])
        np.save(fdir + '/sample-reward', self.reward[:self.size])
        np.save(fdir + '/sample-ndone', self.not_done[:self.size])

    def load(self, fdir):
        state = np.load(fdir + '/sample-state.npy', allow_pickle=True)
        action = np.load(fdir + '/sample-action.npy', allow_pickle=True)
        nstate = np.load(fdir + '/sample-nstate.npy', allow_pickle=True)
        reward = np.load(fdir + '/sample-reward.npy', allow_pickle=True)
        ndone = np.load(fdir + '/sample-ndone.npy', allow_pickle=True)
        for s, a, ns, r, nd in zip(state, action, nstate, reward, ndone):
            self.add(s, a, ns, r, 1. - nd)

    def reset(self):
        self.ptr = 0
        self.size = 0
