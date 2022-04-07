# -*- coding:utf-8 -*- 
import numpy
from define import define
from Performance import Performance
import time

"""
ToM算法

"""


class ToM1 ():
    """
    初始化ToM1
    policy: 策略数
    belief0: 模拟ToM0的belief矩阵
    belief1: ToM1的belief矩阵

    """

    def __init__(self, player_state):
        self.policy = define.policy_num
        self.op_policy = define.policy_num
        self.memory_length = define.memory_length
        self.threshold = 0.65
        # 0表示有球， 1表示无球
        self.has_ball = 0
        # 对手的belief
        # self.belief0 = numpy.repeat(0.25, self.op_policy)
        # 自己的belief
        # self.belief1 = numpy.repeat(0.25, self.policy)
        # self.belief0 = [0.4, 0.3, 0.2, 0.1]
        self.belief0 = numpy.random.rand(define.policy_num)
        self.belief0 = self.belief0[:define.policy_num] / sum(self.belief0)
        self.belief1 = numpy.random.rand(define.policy_num)
        self.belief1 = self.belief1[:define.policy_num] / sum(self.belief1)
        # average initialization
        # self.belief0 = []
        # self.belief1 = []
        # for i in range (self.policy):
        # self.belief0.append (1.0 / self.policy)
        # self.belief1.append (1.0 / self.policy)
        # self.belief0 = [0.4, 0.3, 0.2, 0.1]
        # self.belief1 = [0.4, 0.3, 0.2, 0.1]
        # 自己的performance_model
        self.performance_model = Performance(self.policy, self.op_policy);
        # 对手的performance_model,自己的policy是对手的opPolicy
        self.op_performance_model = Performance(self.op_policy, self.policy);
        self.confidence = define.confidence
        self.pretect_op_policy = 0
        self.memory_time = numpy.zeros(self.memory_length, dtype=float)
        self.memory_pool = numpy.repeat(-2.0, self.memory_length)
        # 用第一层
        self.use_belief1 = numpy.random.randint(2)
        # 是否学习
        self.is_learning = define.is_learning

    def reset(self):
        # random initialization
        # self.belief0 = [0.4, 0.3, 0.2, 0.1]
        self.belief0 = numpy.random.rand(define.policy_num)
        self.belief0 = self.belief0[:define.policy_num] / sum(self.belief0)
        self.belief1 = numpy.random.rand(define.policy_num)
        self.belief1 = self.belief1[:define.policy_num] / sum(self.belief1)
        # average initialization
        #self.belief0 = []
        #self.belief1 = []
        #for i in range (self.policy):
            #self.belief0.append (1.0 / self.policy)
            #self.belief1.append (1.0 / self.policy)
        self.memory_time = numpy.zeros(self.memory_length, dtype=float)
        self.memory_pool = numpy.repeat(-2.0, self.memory_length)
        self.use_belief1 = numpy.random.randint(2)

    """
    预测自己的动作

    """

    def mindBelief(self):
        # 初始化对手max_value min_value
        # print 'belief1'
        op_max_value = numpy.zeros(self.op_policy, dtype=float)
        op_min_value = numpy.zeros(self.op_policy, dtype=float)
        u = numpy.zeros(self.op_policy, dtype=float)
        for i in range(self.op_policy):
            # 计算出平均收益mu
            for j in range(self.policy):
                # 通过op_performance_model预测对手reward
                value = self.op_performance_model.predictReward(i, j)
                op_max_value[i] += value * self.belief1[j]
        for i in range(self.op_policy):
            # 计算出平均收益mu
            for j in range(self.policy):
                # F * Belief
                cdf = self.op_performance_model.predictCdf(max(op_max_value), i, j)
                op_min_value[i] += cdf * self.belief1[j]
        # 预测出对手策略
        # op_policy_index = numpy.argmax(op_max_value)
        op_policy_index = numpy.argmin(op_min_value)
        self.pretect_op_policy = op_policy_index
        # 更新u值
        # self.confidence = 0
        # 检测是否用第一层预测
        self.checkWinTimes()
        # print(self.use_belief1)
        for i in range(self.op_policy):
            if self.use_belief1 == 1:
                if i == op_policy_index:
                    u[i] = (1 - self.confidence) * self.belief0[i] + self.confidence
                else:
                    u[i] = (1 - self.confidence) * self.belief0[i]
            else:
                u[i] = self.belief0[i]
        # 初始化自己max_value min_value
        max_value = numpy.zeros(self.policy, dtype=float)
        min_value = numpy.zeros(self.policy, dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                # 通过performance_model预测自己的reward
                max_value[i] += self.performance_model.predictReward(i, j) * u[j]
        # print "u:", u
        for i in range(self.policy):
            for j in range(self.op_policy):
                # F * Belief
                cdf = self.performance_model.predictCdf(max(max_value), i, j)
                min_value[i] += cdf * u[j]
        # print "min_value:", min_value
        return numpy.argmin(min_value)

    """
    预测自己的动作

    """

    def mindBeliefBayes(self):
        # 初始化max_value
        op_max_value = numpy.zeros(self.op_policy, dtype=float)
        u = numpy.zeros(self.op_policy, dtype=float)
        for i in range(self.op_policy):
            for j in range(self.policy):
                # 通过op_performance_model预测对手reward
                op_max_value[i] += self.op_performance_model.predictReward(i, j) * self.belief1[j]
        # 预测出对手策略
        op_policy_index = numpy.argmax(op_max_value)
        self.pretect_op_policy = op_policy_index
        # 更新u值
        for i in range(self.op_policy):
            if i == op_policy_index:
                u[i] = (1 - self.confidence) * self.belief0[i] + self.confidence
            else:
                u[i] = (1 - self.confidence) * self.belief0[i]
        max_value = numpy.zeros(self.policy, dtype=float)
        for i in range(self.policy):
            for j in range(self.op_policy):
                # 通过performance_model预测自己的reward
                max_value[i] += self.performance_model.predictReward(i, j) * u[j]
        return numpy.argmax(max_value)

    # 更新策略参数
    def updatePolicy(self, policy, op_policy, reward, op_reward):
        # 更新belief1
        # p * belief 根据对手的reward更新belief1
        probability = numpy.zeros(self.policy, dtype=float)
        probability_total = 0.0
        for i in range(self.policy):
            soft_max = self.op_performance_model.predictPdf(op_reward, self.pretect_op_policy, i) * self.belief1[i]
            soft_max = max(soft_max, 0.0001)
            probability[i] = soft_max
            probability_total += probability[i]
        if self.use_belief1 == 1:
            for i in range(self.policy):
                self.belief1[i] = probability[i] / probability_total
        # print "belief1:", self.belief1
        # 更新belief0
        # p * belief 根据自己的reward更新belief0
        op_probability = numpy.zeros(self.op_policy, dtype=float)
        op_probability_total = 0.0
        for i in range(self.op_policy):
            soft_max = self.performance_model.predictPdf(reward, policy, i) * self.belief0[i]
            soft_max = max(soft_max, 0.0001)
            op_probability[i] = soft_max
            op_probability_total += op_probability[i]
        # print op_probability_total
        for i in range(self.op_policy):
            self.belief0[i] = op_probability[i] / op_probability_total
        # 更新confidence
        if self.use_belief1 == 1:
            if reward > op_reward:
                self.confidence = (1 - define.learning_rate) * self.confidence + define.learning_rate
            else:
                self.confidence = (1 - define.learning_rate) * self.confidence
        # print self.confidence
        min_memery = numpy.argmin(self.memory_time)
        self.memory_time[min_memery] = time.time()
        self.memory_pool[min_memery] = reward
        # print self.memory_pool

    def updatePolicyBayes(self, policy, op_policy, reward, op_reward):
        # 更新belief1和confidence
        for i in range (self.policy):
            if i == policy:
                self.belief1[i] = self.belief1[i] * (1 - define.learning_rate) + define.learning_rate
            else:
                self.belief1[i] = self.belief1[i] * (1 - define.learning_rate)
        # 更新belief0
        for i in range (self.op_policy):
            if i == op_policy:
                self.belief0[i] = self.belief0[i] * (1 - define.learning_rate) + define.learning_rate
            else:
                self.belief0[i] = self.belief0[i] * (1 - define.learning_rate)
        if self.pretect_op_policy == op_policy:
            self.confidence = self.confidence * (1 - define.learning_rate) + define.learning_rate
        else:
            self.confidence = self.confidence * (1 - define.learning_rate)

    def updatePerformanceModel(self, policy, op_policy, reward, op_reward):
        # 更新perforamce_model
        self.performance_model.updateParameter(policy, op_policy, reward)
        self.op_performance_model.updateParameter(op_policy, policy, op_reward)

    def getReward(self):
        return 0

    def savePerformanceModel(self):
        numpy.save('tom1_per.npy', self.performance_model.performance_model)
        numpy.save('tom1_op_per.npy', self.op_performance_model.performance_model)

    def loadPerformanceModel(self):
        self.performance_model.performance_model = numpy.load('tom1_per.npy')
        self.op_performance_model.performance_model = numpy.load('tom1_op_per.npy')
        #print(self.performance_model.performance_model)

    # 检测置信度梯度
    def checkConfidence(self):
        if self.memory_pool[len(self.memory_pool) - 1] == -1:
            return True
        gradient = []
        for i in range(1, len(self.memory_pool)):
            gradient.append((self.memory_pool[i] - self.memory_pool[i - 1]) / self.memory_pool[i - 1])
        sum_max0 = 0
        for a in gradient:
            if a >= 0: sum_max0 += 1
        # print gradient
        if sum_max0 * 1.0 / len(gradient) >= 0.9:
            return True
        else:
            return False

        # 检测输赢次数

    def checkWinTimes(self):
        if self.memory_pool[len(self.memory_pool) - 1] == -2:
            return
        times = 0
        for i in range(len(self.memory_pool)):
            if self.memory_pool[i] > 0: times += self.memory_pool[i]
        # print gradient
        # print(self.memory_pool)
        # print("a", times * 1.0 / len(self.memory_pool))
        if times * 1.0 / len(self.memory_pool) >= self.threshold:
            return
        else:
            self.use_belief1 = 1 - self.use_belief1
            self.memory_time = numpy.zeros(self.memory_length, dtype=float)
            self.memory_pool = numpy.repeat(-2.0, self.memory_length)
        return

    def getIsLearning(self):
        return self.is_learning

