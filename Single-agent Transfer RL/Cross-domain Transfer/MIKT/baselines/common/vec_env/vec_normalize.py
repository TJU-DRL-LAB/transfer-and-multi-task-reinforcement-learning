from . import VecEnvWrapper
from baselines.common.running_mean_std import RunningMeanStd, TfRunningMeanStd
import numpy as np
import pickle


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10.,
                 gamma=0.99, epsilon=1e-8, mean_path=None, ret_path=None,
                 load_mean_path=None, load_ret_path=None, transfer=None,
                 transfer_env=None):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        if load_mean_path:
            with open(load_mean_path, 'rb') as f:
                self.ob_rms = pickle.load(f)
        if load_ret_path:
            with open(load_ret_path, 'rb') as f:
                self.ret_rms = pickle.load(f)

        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.mean_path = mean_path
        self.ret_path = ret_path

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def save(self):
        if self.mean_path:
            with open(self.mean_path, 'wb') as filehandler:
                pickle.dump(self.ob_rms, filehandler, pickle.HIGHEST_PROTOCOL)
        if self.ret_path:
            with open(self.ret_path, 'wb') as filehandler:
                pickle.dump(self.ret_rms, filehandler, pickle.HIGHEST_PROTOCOL)
