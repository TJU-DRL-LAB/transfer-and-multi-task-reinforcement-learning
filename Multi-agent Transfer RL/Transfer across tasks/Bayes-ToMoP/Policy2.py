# -*- coding:utf-8 -*- 
from BallGame import BallGame
import numpy as np
import math
from define import define
import random


class Policy2():
    def __init__(self):
        # 0代表带球直接冲球门 不带球直接去抢球
        # 1代表带球直接冲球门 不带球绕道去抢球
        # 2代表带球绕道冲球门 不带球直接去抢球
        # 3代表带球绕道冲球门 不带球绕道去抢球
        self.policy = 4
        self.action_num = 4
        # 动作
        self.left = 0
        self.up = 1
        self.right = 2
        self.down = 3
        self.ball_game = BallGame()

    # policy_index: 策略类型  player_state：是A或者是B location:自己位置  op_location：对手位置 has_ball:0表示带球1表示不带球
    def getActionMatrix(self, policy_index, player_state, location, op_location, has_ball):
        if (policy_index == 0 or policy_index == 1) and has_ball == 0: return self.doPolicy0(player_state)                      # 带球积极
        elif (policy_index == 2 or policy_index == 3) and has_ball == 0: return self.doPolicy1(player_state, location)       # 带球消极
        elif (policy_index == 0 or policy_index == 2) and has_ball == 1: return self.doPolicy2(player_state)       # 不带球积极
        elif (policy_index == 1 or policy_index == 3) and has_ball == 1: return self.doPolicy3(player_state, location)                    # 不带球消极
        elif (policy_index == 4) and has_ball == 0:
            return self.doPolicy4(player_state, location)
        elif (policy_index == 4) and has_ball == 1:
            return self.doPolicy2(player_state)

    """
    带球直接冲球门

    """
    def doPolicy0(self, player_state):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        if player_state == 0:
            action_maxtrix[2] = 1
        elif player_state == 1:
            action_maxtrix[0] = 1
        return action_maxtrix

    """
    带球绕道冲球门

    """
    def doPolicy1(self, player_state, location):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        # 在起始点
        if player_state == 0 and location == self.ball_game.getStartA():
            action_maxtrix[3] = 1
        elif player_state == 0 and location != self.ball_game.getStartA():
            action_maxtrix[2] = 1
        elif player_state == 1 and location == self.ball_game.getStartB():
            action_maxtrix[3] = 1
        elif player_state == 1 and location != self.ball_game.getStartB():
            action_maxtrix[0] = 1
        #print next_distance
        return action_maxtrix

    """
    不带球直接去抢球

    """
    def doPolicy2(self, player_state):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        if player_state == 0:
            action_maxtrix[2] = 1
        elif player_state == 1:
            action_maxtrix[0] = 1
        #print next_distance
        return action_maxtrix

    """
    不带球绕道去抢球

    """
    def doPolicy3(self, player_state, location):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        # 在起始点
        if player_state == 0 and location == self.ball_game.getStartA():
            action_maxtrix[3] = 1
        elif player_state == 0 and location != self.ball_game.getStartA():
            action_maxtrix[2] = 1
        elif player_state == 1 and location == self.ball_game.getStartB():
            action_maxtrix[3] = 1
        elif player_state == 1 and location != self.ball_game.getStartB():
            action_maxtrix[0] = 1
        #print next_distance
        return action_maxtrix

    # 新策略
    def doPolicy4(self, player_state, location):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        # 在起始点
        if player_state == 0 and location == self.ball_game.getStartA():
            action_maxtrix[1] = 1
        elif player_state == 0 and location == [5, 0]:
            action_maxtrix[3] = 1
        elif player_state == 0 and location != self.ball_game.getStartA():
            action_maxtrix[2] = 1
        elif player_state == 1 and location == self.ball_game.getStartB():
            action_maxtrix[1] = 1
        elif player_state == 1 and location == [1, 0]:
            action_maxtrix[3] = 1
        elif player_state == 1 and location != self.ball_game.getStartB():
            action_maxtrix[0] = 1
        # print next_distance
        return action_maxtrix



