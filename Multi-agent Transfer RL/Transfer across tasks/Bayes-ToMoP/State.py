# -*- coding:utf-8 -*-
import numpy
from define import define
from BallGame import BallGame
import random
class State(object):
    # State 横纵编号
    def __init__(self, row, column):
        # model-based V(s), 0有球，1无球
        self.action_num = 4
        self.V = [0, 0]
        self.Q = numpy.zeros(self.action_num, dtype=float)
        self.Q1 = numpy.zeros(self.action_num, dtype=float)
        self.maxAction = [0, 0]
        self.total_R = numpy.zeros(self.action_num, dtype=float)
        self.n = numpy.zeros(self.action_num, dtype=float)
        # 状态转移矩阵，存储概率和转移状态
        self.T = [[[1.0] * 1, [0, 0] * 1] for i in range(self.action_num)]
        self.row = row
        self.column = column
        self.greedy = define.greedy
        self.learning = define.learning
        # 初始化转移矩阵T
        self.player_state = 0
        self.BallGame = BallGame()
        self.InitT()
        
        

    def InitT(self):
        for i in range(self.action_num):
            # 左
            if i == 0:
                self.T[i][1][0] = self.row - 1
                self.T[i][1][1] = self.column
            # 上
            elif i == 1:
                self.T[i][1][0] = self.row
                self.T[i][1][1] = self.column - 1
            # 右
            elif i == 2:
                self.T[i][1][0] = self.row + 1
                self.T[i][1][1] = self.column
            # 下
            elif i == 3:
                self.T[i][1][0] = self.row
                self.T[i][1][1] = self.column + 1
        while(True):
            action = random.randint(0, self.action_num - 1)
            if(self.IsLegalAction(self.player_state, self.T[action][1])):
                self.maxAction = [action, action]
                break

    def chooseAction(self, player_state, has_ball, greedy):
        rand = random.random()
        position = [self.row, self.column]
        # 随机选动作
        if rand < greedy:
            while(True):
                action = random.randint(0, self.action_num - 1)
                next_position = self.T[action][1]
                if(self.IsLegalAction(player_state, next_position)):
                    return action
        # 选MaxQ最大的
        else:
            return self.maxAction[has_ball]

    def chooseActionQ(self, player_state, has_ball, greedy):
        rand = random.random()
        position = [self.row, self.column]
        # 随机选动作
        if rand < self.greedy:
            while(True):
                action = random.randint(0, self.action_num - 1)
                next_position = self.T[action][1]
                if(self.IsLegalAction(player_state, next_position)):
                    return action
        # 选MaxQ最大的
        else:
            maxQ = 0
            maxA = 0
            for i in range(self.action_num):
                if self.IsLegalAction(player_state, self.T[i][1]) and has_ball == 0 and maxQ < self.Q[i]:
                    maxQ = self.Q[i]
                if self.IsLegalAction(player_state, self.T[i][1]) and has_ball == 1 and maxQ < self.Q1[i]:
                    maxQ = self.Q1[i]
            while(True):
                action = random.randint(0, self.action_num - 1)
                if has_ball == 0 and maxQ == self.Q[action]:
                    return action
                if has_ball == 1 and maxQ == self.Q1[action]:
                    return action

    def getMaxQ(self, has_ball):
        maxQ = 0
        for i in range(self.action_num):
            if self.IsLegalAction(0, self.T[i][1]) and has_ball == 0 and maxQ < self.Q[i]:
                maxQ = self.Q[i]
            if self.IsLegalAction(0, self.T[i][1]) and has_ball == 1 and maxQ < self.Q1[i]:
                maxQ = self.Q1[i]
        #print self.Q,self.Q1
        return maxQ

    def IsLegalAction(self, player_state, position):
        # 球门区域
        if(player_state == 0 and (position in self.BallGame.goal_B)):
            return True
        elif(player_state == 1 and (position in self.BallGame.goal_A)):
            return True
        # 球门以外的区域
        if(position[0] < 1 or position[0] >= self.BallGame.game_width - 1):
            return False
        if(position[1] < 0 or position[1] >= self.BallGame.game_height):
            return False
        return True

    def updateReward(self, reward, action):
        self.total_R[action] += reward
        self.n[action] += 1
        self.greedy = self.greedy * (1 - define.greedy_discount)
        self.learning = self.learning * (1 - define.learning_discount)

    def getR(self, action):
        if(self.n[action] == 0):
            return 0
        return self.total_R[action] / self.n[action]

    def getNextState(self, action):
        return self.T[action][1]

    def getV(self, has_ball):
        #print self.V[has_ball]
        return self.V[has_ball]

    def setV(self, has_ball, v):
        self.V[has_ball] = v

    def setMaxAction(self, has_ball, action):
        self.maxAction[has_ball] = action
