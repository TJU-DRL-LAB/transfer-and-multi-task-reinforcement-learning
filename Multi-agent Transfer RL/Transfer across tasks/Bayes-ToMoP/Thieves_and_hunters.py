# -*- coding:utf-8 -*- 
import numpy as np
import random
import math
"""
Thieves_and_hunters游戏

"""


class Thieves_and_hunters():
    def __init__(self):
        # game size , Add a circle of boundaries
        self.game_width = 9
        self.game_height = 9
        # A and B starts area
        self.start_A = [[0, 4]]
        self.start_B = [[self.game_width - 1, 4]]
        # the goals of A  and B
        self.goals = []
        self.goals_score = []
        self.catch_goals_num_A = 0
        self.catch_goals_num_B = 0
        self.A_score = 0
        self.B_score = 0
        self.A_position = [0, 0]
        self.B_position = [0, 0]
        self.game_grid = []
        self.init()
        self.action_num = 5
    
    def init(self):
        self.initGrid()
        self.resetScore()

    # init game grid, 初始化时，每一个格子都是不可到达的。
    # #表示不可达到，A表示球员A可以到达，B表示球员B可以到达，O表示都可以到达，G表示目标点
    def initGrid(self):
        self.game_grid = []
        for i in range(self.game_height):
            list = []
            for j in range(self.game_width):
                list.append('#')
            self.game_grid.append(list)

        self.initAArea()
        self.initBArea()
        self.setGoals()
        self.setStartA()
        self.setStartB()

    # 设置A能到达的格子
    def initAArea(self):
        for i in range(self.game_height - 2):
            for j in range(3):
                self.game_grid[i + 1][j + 1] = 'A'

    # 设置B能到达的格子
    def initBArea(self):
        for i in range(self.game_height - 2):
            for j in range(3):
                self.game_grid[i + 1][j + 5] = 'B'
                
    def setGoals(self):
        self.goals = []
        self.goals_score = []
        self.goals.append([4, 1])
        self.goals_score.append(1)
        self.goals.append([4, 3])
        self.goals_score.append(1)
        self.goals.append([4, 5])
        self.goals_score.append(1)
        self.goals.append([4, 7])
        self.goals_score.append(1)

        self.game_grid[1][4] = 'G'
        self.game_grid[3][4] = 'G'
        self.game_grid[5][4] = 'G'
        self.game_grid[7][4] = 'G'

    def setStartA(self):
        self.A_position[0] = self.start_A[0][0]
        self.A_position[1] = self.start_A[0][1]
        self.game_grid[self.A_position[1]][self.A_position[0]] = 'A'

    def setStartB(self):
        self.B_position[0] = self.start_B[0][0]
        self.B_position[1] = self.start_B[0][1]
        self.game_grid[self.B_position[1]][self.B_position[0]] = 'B'

    def printGrid(self):
        for i in range(self.game_height):
            list = []
            for j in range(self.game_width):
                list.append(self.game_grid[i][j])
            print(list)

    def resetScore(self):
        self.A_score = 0
        self.B_score = 0
        self.catch_goals_num_A = 0
        self.catch_goals_num_B = 0

    def endGame(self):
        total = np.sum(self.goals_score)
        if total == 0:
            return True
        else:
            return False

    # action: 0停，1左，2上，3右，4下
    def move(self, action, player):
        position = [0, 0]
        if player == 'A':
            position[0] = self.A_position[0]
            position[1] = self.A_position[1]
        elif player == 'B':
            position[0] = self.B_position[0]
            position[1] = self.B_position[1]

        next_position = [0, 0]
        if action == 5:
            next_position[0] = position[0]
            next_position[1] = position[1]
        elif action == 0:
            next_position[0] = position[0] - 1
            next_position[1] = position[1]
        elif action == 1:
            next_position[0] = position[0]
            next_position[1] = position[1] - 1
        elif action == 2:
            next_position[0] = position[0] + 1
            next_position[1] = position[1]
        elif action == 3:
            next_position[0] = position[0]
            next_position[1] = position[1] + 1
        if self.IsOutOfBound(next_position, player):
            return -1
        
        if player == 'A':
            self.A_position[0] = next_position[0]
            self.A_position[1] = next_position[1]
        elif player == 'B':
            self.B_position[0] = next_position[0]
            self.B_position[1] = next_position[1]
        return next_position

    def check(self):
        self.IsScore()
        self.inGoals()

    def LegalAction(self, action, player):
        position = [0, 0]
        if player == 'A':
            position[0] = self.A_position[0]
            position[1] = self.A_position[1]
        elif player == 'B':
            position[0] = self.B_position[0]
            position[1] = self.B_position[1]
        next_position = [0, 0]
        if action == 5:
            next_position[0] = position[0]
            next_position[1] = position[1]
        elif action == 0:
            next_position[0] = position[0] - 1
            next_position[1] = position[1]
        elif action == 1:
            next_position[0] = position[0]
            next_position[1] = position[1] - 1
        elif action == 2:
            next_position[0] = position[0] + 1
            next_position[1] = position[1]
        elif action == 3:
            next_position[0] = position[0]
            next_position[1] = position[1] + 1
        if self.IsOutOfBound(next_position, player):
            return -1
        else:
            return next_position

    def distance(self, position, goals_index):
        if position in self.goals and position != self.goals[goals_index]:
            return 100
        return math.fabs(position[0] - self.goals[goals_index][0]) + math.fabs(position[1] - self.goals[goals_index][1])

    def BeCatched(self):
        if self.A_position == self.B_position:
            self.B_score -= 1
            self.A_score += 1
            if self.A_position in self.goals:
                index = self.goals.index(self.A_position)
                self.game_grid[self.A_position[1]][self.A_position[0]] = 'O'
                self.goals_score[index] = 0
            return True
        return False

    def IsScore(self):
        if self.BeCatched():
            return False
        if self.A_position not in self.goals:
            return False
        index = self.goals.index(self.A_position)
        self.A_score += self.goals_score[index]
        self.B_score -= self.goals_score[index]
        self.game_grid[self.A_position[1]][self.A_position[0]] = 'O'
        self.goals_score[index] = 0
        return True
    
    def inGoals(self):
        if self.A_position in self.goals:
            index = self.goals.index(self.A_position)
            self.catch_goals_num_A += 1
        if self.B_position in self.goals:
            self.catch_goals_num_B += 1

    def getStartA(self):
        return self.start_A[0]

    def getStartB(self):
        return self.start_B[0]

    def IsOutOfBound(self, position, player):
        if position[0] < 0 or position[0] > 8:
            return True
        if self.game_grid[position[1]][position[0]] == player or self.game_grid[position[1]][position[0]] == 'O' or self.game_grid[position[1]][position[0]] == 'G':
            return False
        return True

    def chooseAction(self, goals_index, player):
        # 获取当前状态下动作概率矩阵
        action_maxtrix = self.getActionMatrix(goals_index, player)
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
            if rand <= i:
                return index
            index += 1
        return index

    def getActionMatrix(self, goals_index, player):
        action_matrix = []
        for i in range(self.action_num):
            action_matrix.append(0)
        if player == 'A':
            position = self.A_position
        else:
            position = self.B_position
        distance = self.distance(position, goals_index)
        while True:
            rand = random.randint(0, self.action_num - 2)
            next_position = self.LegalAction(rand, player)
            if next_position == -1:
                continue
            if self.distance(next_position, goals_index) <= distance:
                action_matrix[rand] = 1
                break
        return action_matrix
