# -*- coding:utf-8 -*- 
import numpy as np
import random
from scipy.stats import norm


class Performance():
    def __init__(self, policy, op_policy):
        # BallGame里，一共有4个策略，0代表持球积极策略，1代表持球不积极策略，2代表不持球积极策略，3代表不持球不积极策略
        # policy 表示自己的策略个数，op_policy表示对手策略个数
        self.policy = policy
        self.op_policy = op_policy
        # 样本个数
        self.count = np.zeros((self.policy, self.op_policy), dtype=float)
        # 每个策略的高斯分布参数（mu, sigma）
        self.performance_model = np.zeros((self.policy, self.op_policy, 2), dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                self.performance_model[i, j, 1] = 1

    def updateParameter(self, policy_index, op_policy_index, value):
        if self.count[policy_index, op_policy_index] == 0:
            self.performance_model[policy_index, op_policy_index, 0] = value

        else:
            # 均值
            mean = self.performance_model[policy_index, op_policy_index, 0]
            count = self.count[policy_index, op_policy_index]
            self.performance_model[policy_index, op_policy_index, 0] = (value - mean) / (count + 1.0) + mean
            # 方差
            sigma = self.performance_model[policy_index, op_policy_index, 1]
            self.performance_model[policy_index, op_policy_index, 1] = (count - 1.0) / count * sigma + (value - mean) ** 2 / (count + 1.0)
            if self.performance_model[policy_index, op_policy_index, 1] < 0.01:
                self.performance_model[policy_index, op_policy_index, 1] = 0.01
        # 样本个数
        self.count[policy_index, op_policy_index] += 1.0

    def predictReward(self, policy_index, op_policy_index):
        #rand = random.gauss(self.performance_model[policy_index, op_policy_index, 0], self.performance_model[policy_index, op_policy_index, 1])
        return self.performance_model[policy_index, op_policy_index, 0]

    def predictCdf(self, value, policy_index, op_policy_index):
        return norm.cdf(value, self.performance_model[policy_index, op_policy_index, 0], self.performance_model[policy_index, op_policy_index, 1])

    def predictPdf(self, value, policy_index, op_policy_index):
        a = norm.pdf(value, self.performance_model[policy_index, op_policy_index, 0], self.performance_model[policy_index, op_policy_index, 1])
       # print a, value, self.performance_model[policy_index, op_policy_index, 0], self.performance_model[policy_index, op_policy_index, 1], "aaaaaaaaaaaaaaaaaaaaaaaaaa"
        return a

