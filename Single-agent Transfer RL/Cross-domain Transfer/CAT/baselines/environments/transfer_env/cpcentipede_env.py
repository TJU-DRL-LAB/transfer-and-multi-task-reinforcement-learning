import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import baselines.environments.init_path as init_path
import os
import num2words
from baselines.environments.transfer_env.centipede_env import CentipedeEnv

class CpCentipedeFourEnv(CentipedeEnv):

    def __init__(self):
        super(CentipedeFourEnv, self).__init__(CentipedeLegNum=4, is_crippled=True)


class CpCentipedeSixEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6, is_crippled=True)


class CpCentipedeEightEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8, is_crippled=True)


class CpCentipedeTenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10, is_crippled=True)


class CpCentipedeTwelveEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12, is_crippled=True)


class CpCentipedeFourteenEnv(CentipedeEnv):

    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14, is_crippled=True)
