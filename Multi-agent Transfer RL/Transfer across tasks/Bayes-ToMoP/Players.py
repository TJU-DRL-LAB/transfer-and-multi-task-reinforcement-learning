# -*- coding:utf-8 -*- 
from BallGame import BallGame
from Policy import Policy
from Policy2 import Policy2
from ToM1 import ToM1
from ToM0 import ToM0
import random
from PolicyTF import PolicyTF
from ModelBased import ModelBased
"""
踢球游戏选手类型

"""


class Player():
    def __init__(self, player_state):
        # 0表示A, 1表示B
        self.player_state = player_state

        # 0表示有球， 1表示无球
        self.has_ball = 0
        # 动作
        self.left = 0
        self.up = 1
        self.right = 2
        self.down = 3
        self.stay = 4
        self.action_num = 5
        # 踢球游戏矩阵
        self.BallGame = BallGame()
        self.location = [0, 0]
        self.op_location = [0, 0]
        # 选择的策略
        self.policy_index = 0
        self.policy = Policy()
        self.policy2 = Policy2()
        self.PolicyTF = PolicyTF()
        self.Model_Based = ModelBased()
        self.has_ball_cur = 0
        if player_state == 0:
            # left: 0(self) right: 1(opponent)
            self.tom = ToM1(player_state)
        else:
            self.tom = ToM0(player_state)

    # 策略个数
    def getPolicyNum(self):
        return self.policy.policy
        #return self.PolicyTF.policy

    def getpretect(self):
        return self.tom.pretect_op_policy

    # 设置策略
    def setPolicyIndex(self, policy_index):
        self.policy_index = policy_index

    # 设置球权
    def setRightToServe(self, ball_right):
        self.has_ball = ball_right

    # 设置初始位置
    def setLocation(self):
        if self.player_state == 0:
            loc = self.BallGame.getStartA()
            self.location[0] = loc[0]
            self.location[1] = loc[1]
        else:
            loc = self.BallGame.getStartB()
            self.location[0] = loc[0]
            self.location[1] = loc[1]

    # 设置对手初始位置
    def setOpLocation(self, op_location):
        self.op_location = op_location

    # 获取初始位置
    def getLocation(self):
        return self.location

    # 获取球权
    def getBallRight(self):
        return self.has_ball

    # 选择策略
    def choosePolicy(self):
        self.policy_index = self.tom.mindBelief()
        return self.policy_index

    def choosePolicyBayes(self):
        self.policy_index = self.tom.mindBeliefBayes()
        return self.policy_index

    # 更新performance_model
    def updatePerformanceModel(self, policy, op_policy, reward, op_reward):
        self.tom.updatePerformanceModel(policy, op_policy, reward, op_reward)

    # 保存performance_model
    def savePerformanceModel(self):
        self.tom.savePerformanceModel()

    # 加载performance_model
    def loadPerformanceModel(self):
        self.tom.loadPerformanceModel()

    # 更新ToM参数
    def updatePolicy(self, policy, op_policy, reward, op_reward):
        self.tom.updatePolicy(policy, op_policy, reward, op_reward)

    def updatePolicyBayes(self, policy, op_policy, reward, op_reward):
        self.tom.updatePolicyBayes(policy, op_policy, reward, op_reward)

    def chooseAction(self):
        # 启用学习机制
        if self.player_state == 0 and self.tom.getIsLearning():
            return self.Model_Based.getAction(self.location, self.player_state, self.has_ball_cur)

        # 获取当前状态下动作概率矩阵
        action_maxtrix = self.policy2.getActionMatrix(self.policy_index, self.player_state, self.location, self.op_location, self.has_ball)
        #print action_maxtrix
        # 将概率矩阵分布在0-1之间
        rand_total = 0.0
        index = 0
        for i in action_maxtrix:
            action_maxtrix[index] = rand_total + i
            rand_total += i
            index += 1
        # 选择动作
        rand = random.random()
        index = 0
        for i in action_maxtrix:
            if(rand <= i):
                return index
            index += 1
        return index

    """
    选手移动
    """
    def move(self, action):
        if action == self.left:
            if self.IsLegalAction(self.location[0] - 1, self.location[1]):
                self.location[0] = self.location[0] - 1
        elif action == self.right:
            if self.IsLegalAction(self.location[0] + 1, self.location[1]):
                self.location[0] = self.location[0] + 1
        elif action == self.up:
            if self.IsLegalAction(self.location[0], self.location[1] - 1):
                self.location[1] = self.location[1] - 1
        elif action == self.down:
            if self.IsLegalAction(self.location[0], self.location[1] + 1):
                self.location[1] = self.location[1] + 1
        return self.location

    def IsScored(self):
        #print self.has_ball, self.player_state, self.location
        # 有球 如果是A
        if self.has_ball == 0 and self.player_state == 0 and self.location in self.BallGame.goal_B:
            return True
        if self.has_ball == 0 and self.player_state == 1 and self.location in self.BallGame.goal_A:
            return True
        return False

    """
    判断是否是合理的踢球

    """
    def IsLegalAction(self, positionX, positionY):
        # 球门区域
        if self.player_state == 0 and ([positionX, positionY] in self.BallGame.goal_B):
            return True
        elif self.player_state == 1 and ([positionX, positionY] in self.BallGame.goal_A):
            return True
        if positionX < 1 or positionX >= self.BallGame.game_width - 1:
            return False
        if positionY < 0 or positionY >= self.BallGame.game_height:
            return False
        return True

    def getGoalsIndex(self, catch_goals_num):
        return self.PolicyTF.getGoals(self.policy_index, catch_goals_num) - 1
