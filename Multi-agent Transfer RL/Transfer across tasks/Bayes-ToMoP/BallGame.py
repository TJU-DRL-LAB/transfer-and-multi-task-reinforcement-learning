# -*- coding:utf-8 -*- 
import numpy as np
import random
"""
踢球游戏

"""


class BallGame():
    def __init__(self):
        # game size
        self.game_width = 7
        self.game_height = 4
        # A and B starts area
        self.start_A = []
        self.start_B = []
        for i in range(self.game_height):
            self.start_A.append([1, i])
            self.start_B.append([self.game_width - 2, i])
        # the goals of A  and B
        self.goal_A = [[0, 1], [0, 2]]
        self.goal_B = [[self.game_width - 1, 1], [self.game_width - 1, 2]]

    def getStartA(self):
        rand = random.randint(0, self.game_height - 1)
        return self.start_A[1]

    def getStartB(self):
        rand = random.randint(0, self.game_height - 1)
        return self.start_B[1]
