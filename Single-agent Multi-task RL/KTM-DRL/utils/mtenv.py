import os

import gym
import numpy as np
from tensorboardX import SummaryWriter

from agent.td3 import TD3
from utils.replay import ReplayBuffer


class MTEnv:
    def __init__(self, seed):
        self.env_name_list = []
        self.env_list = []
        self.env_test_list = []
        self.state_dim_list = []
        self.action_dim_list = []
        self.max_action_list = []

        self.seed = seed

        self.state_dim = 0
        self.action_dim = 0

    def get_env_idx(self, name):
        return self.env_name_list.index(name)

    def get_env(self, idx):
        return self.env_list[idx]

    def get_env_test(self, idx):
        return self.env_test_list[idx]

    def get_env_by_name(self, name):
        return self.env_list[self.get_env_idx(name)]

    def get_env_name_by_idx(self, idx):
        return self.env_name_list[idx]

    def add_env(self, env_name):
        self.env_name_list.append(env_name)
        env = gym.make(env_name)
        env_test = gym.make(env_name)

        env.seed(self.seed)
        env_test.seed(self.seed + 100)

        s = env.observation_space.shape[0]
        a = env.action_space.shape[0]
        ma = env.action_space.high[0]
        self.state_dim_list.append(s)
        self.action_dim_list.append(a)
        self.max_action_list.append(ma)

        self.env_list.append(env)
        self.env_test_list.append(env_test)

        self.state_dim = np.max(self.state_dim_list)
        self.action_dim = np.max(self.action_dim_list)

        return self.get_env_idx(env_name)

    def step(self, idx, action):
        reward, done, _ = self.env_list[idx].step(action)

    def pad_state(self, state):
        return np.concatenate([state, np.zeros(self.state_dim - len(state))])

    def pad_action(self, action):
        return np.concatenate([action, np.zeros(self.action_dim - len(action))])

    def clip_action(self, idx, action):
        return action[:self.action_dim_list[idx]]


class MTTeacher:
    def __init__(self, teacher_name_list, dir_base, dir_sum, seed, load_buffer=True):
        env_name_list = []
        self.policy_list = []
        self.replay_list = []
        self.summer_list = []
        self.mtenv = MTEnv(seed)

        for tname in teacher_name_list:
            print('Processing', tname)

            env_name = str(tname).split('/')[1]
            env_name_list.append(env_name)

            idx = self.mtenv.add_env(env_name)
            state_dim, action_dim = self.mtenv.state_dim_list[idx], self.mtenv.action_dim_list[idx]

            r = ReplayBuffer(state_dim, action_dim)
            if load_buffer:
                r.load(os.path.join(dir_base, tname))
            self.replay_list.append(r)

            s = SummaryWriter(os.path.join(dir_sum, env_name), flush_secs=10)
            self.summer_list.append(s)

            p = TD3(state_dim, action_dim, 1.0)
            p.load(os.path.join(dir_base, tname, 'mod', 'td3-1000000'))
            self.policy_list.append(p)

    def get_bundle(self, idx):
        return self.policy_list[idx], self.mtenv.get_env(idx), self.replay_list[idx], self.summer_list[idx]

    def get_teacher(self, idx):
        return self.policy_list[idx]

    def reset_all_replay(self):
        for replay in self.replay_list:
            replay.reset()

    def __len__(self):
        return len(self.policy_list)
