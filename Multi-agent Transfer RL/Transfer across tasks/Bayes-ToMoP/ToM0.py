# -*- coding:utf-8 -*- 
import numpy
from define import define
from Performance import Performance

"""
ToM0算法

"""


class ToM0():
    """
    初始化ToM0
    policy: 策略数
    belief0: ToM0的belief矩阵

    """
    def __init__(self, player_state):
        self.player_state = player_state
        self.policy = define.policy_num
        self.op_policy = define.policy_num
        # 对手的belief
        # random initialization
        self.belief0 = numpy.random.rand(define.policy_num)
        self.belief0 = self.belief0[:define.policy_num] / sum(self.belief0)
        # average initialization
        #self.belief0 = []
        #for i in range(self.policy):
            #self.belief0.append(1.0 / self.policy)
        # self.belief0 = [0.1, 0.2, 0.3, 0.4]
        # 自己的performance_model
        self.performance_model = Performance(self.policy, self.op_policy)

    def reset(self):
        # random initialization
        self.belief0 = numpy.random.rand(define.policy_num)
        self.belief0 = self.belief0[:define.policy_num] / sum(self.belief0)
        # average initialization
        #self.belief0 = []
        #for i in range (self.policy):
            #self.belief0.append (1.0 / self.policy)
        # self.belief0 = nu mpy.repeat(0.25, self.op_policy)

        # self.belief0 = [0.1, 0.2, 0.3, 0.4]
    """
    ToM0下选择策略
    
    """
    def mindBelief(self):
        # 初始化策略max_value
        #print 'belief0'
        max_value = numpy.zeros(self.policy, dtype=float)
        min_value = numpy.zeros(self.policy, dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                # 双方策略获得的reward（改）
                value = self.performance_model.predictReward(i, j)
                max_value[i] += self.belief0[j] * value
        # 去最大的value进行F*Belief计算
        for i in range(self.policy):
            for j in range(self.op_policy):
                value = self.performance_model.predictCdf(max(max_value), i, j)
                min_value[i] += self.belief0[j] * value
        #print max_value, min_value
        # 更改belief0矩阵
        return numpy.argmin(min_value)

    def mindBeliefBayes(self):
        max_value = numpy.zeros(self.policy, dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                # 双方策略获得的reward（改）
                max_value[i] += self.belief0[j] * self.performance_model.predictReward(i, j)
        # 更改belief0矩阵
        return numpy.argmax(max_value)

    # 更新策略参数
    def updatePolicy(self, policy, op_policy, reward, op_reward):
        # 更新belief0
        # p * belief 根据对手的reward更新belief0
        probability = numpy.zeros(self.op_policy, dtype=float)
        probability_total = 0.0
        for j in range(self.op_policy):
            soft_max = self.performance_model.predictPdf(reward, policy, j) * self.belief0[j]
            soft_max = max(soft_max, 0.0001)
            probability[j] = soft_max
            probability_total += probability[j]
        for j in range(self.op_policy):
            self.belief0[j] = probability[j] / probability_total

    # 更新策略参数
    def updatePolicyBayes(self, policy, op_policy, reward, op_reward):
        # 更新belief0
        for i in range(self.op_policy):
            if i == op_policy: self.belief0[i] = self.belief0[i] * (1 - define.learning_rate) + define.learning_rate  
            else: self.belief0[i] = self.belief0[i] * (1 - define.learning_rate)

    def updatePerformanceModel(self, policy, op_policy, reward, op_reward):
        # 更新perforamce_model
        self.performance_model.updateParameter(policy, op_policy, reward)

    def savePerformanceModel(self):
        numpy.save('tom0_per.npy', self.performance_model.performance_model)

    def loadPerformanceModel(self):
        if self.player_state == 1:  # left
            self.performance_model.performance_model = numpy.load('tom0_per.npy')
        else:  # right
            self.performance_model.performance_model = numpy.load('tom1_per.npy')
        #print(self.performance_model.performance_model)
