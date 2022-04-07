# -*- coding:utf-8 -*- 
from BallGame import BallGame
import numpy as np
import math
from define import define
import random


class Policy():
    def __init__(self):
        # 0代表带球积极不带球积极
        # 1代表带球积极不带球消极
        # 2代表带球消极不带球积极
        # 3代表带球消极不带球消极
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
        if (policy_index == 0 or policy_index == 1) and has_ball == 0: return self.doPolicy0(player_state, location)                      # 带球积极
        elif (policy_index == 2 or policy_index == 3) and has_ball == 0: return self.doPolicy1(player_state, location, op_location)       # 带球消极
        elif (policy_index == 0 or policy_index == 2) and has_ball == 1: return self.doPolicy2(player_state, location, op_location)       # 不带球积极
        elif (policy_index == 1 or policy_index == 3) and has_ball == 1: return self.doPolicy3(player_state, location)                    # 不带球消极



    """
    持球进攻球门

    """
    def doPolicy0(self, player_state, location):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        legal_action = np.zeros(self.action_num, dtype=float)
        next_distance = np.zeros(self.action_num, dtype=float)
        now_distance = self.getGoalDistance(location, player_state)
        # 符合要求的动作个数
        legal_action_total = 0.0
        # 先找出合格的动作
        for i in range(self.action_num):
            next_location = self.getNextPosition(i, player_state, location)
            if(next_location != location): 
                legal_action[i] = 1
                next_distance[i] = self.getGoalDistance(next_location, player_state)
                legal_action_total += 1
            #else: next_distance[i] = now_distance
        # 求每个动作的概率
        near_goal_total = 0.0
        for i in range(self.action_num):
            # 先将离球门近的动作的值变大，从而求得更大的概率
            if legal_action[i] != 0:
                next_distance[i] = 1.0 / next_distance[i]
                near_goal_total += next_distance[i]
        # 求离球门近的动作概率
        for i in range(self.action_num):
            if legal_action[i] != 0: action_maxtrix[i] = (1 - define.action_rate) * (1 / legal_action_total) + define.action_rate * next_distance[i] / near_goal_total

        action_maxtrix = np.zeros(self.action_num, dtype=float)
        max_dis = np.max(next_distance)
        while(True):
            rand = random.randint(0, 3)
            if legal_action[rand] != 0 and max_dis == next_distance[rand]:
                action_maxtrix[rand] = 1
                break
        #print next_distance
        return action_maxtrix
          
    """
    持球远离对方

    """
    def doPolicy1(self, player_state, location, op_location):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        legal_action = np.zeros(self.action_num, dtype=float)
        next_distance = np.zeros(self.action_num, dtype=float)
        now_distance = self.getOpDistance(location, op_location)
        # 符合要求的动作个数
        legal_action_total = 0.0
        # 先找出合格的动作
        for i in range(self.action_num):
            next_location = self.getNextPosition(i, player_state, location)
            if(next_location != location): 
                legal_action[i] = 1
                next_distance[i] = self.getOpDistance(next_location, op_location)
                legal_action_total += 1
            #else: next_distance[i] = now_distance
        # 求每个动作的概率 离对手距离越近动作概率越小
        near_goal_total = 0.0
        for i in range(self.action_num):
            if legal_action[i] != 0:
                near_goal_total += next_distance[i]
        for i in range(self.action_num):
            if legal_action[i] != 0: action_maxtrix[i] = (1 - define.action_rate) * (1 / legal_action_total) + define.action_rate * next_distance[i] / near_goal_total
        
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        max_dis = np.max(next_distance)
        while(True):
            rand = random.randint(0, 3)
            if legal_action[rand] != 0 and max_dis == next_distance[rand]:
                action_maxtrix[rand] = 1
                break
        #print next_distance 
        return action_maxtrix

    """
    不持球靠近对方

    """
    def doPolicy2(self, player_state, location, op_location):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        legal_action = np.zeros(self.action_num, dtype=float)
        next_distance = np.zeros(self.action_num, dtype=float)
        now_distance = self.getOpDistance(location, op_location)
        # 符合要求的动作个数
        legal_action_total = 0.0
        # 先找出合格的动作
        for i in range(self.action_num):
            next_location = self.getNextPosition(i, player_state, location)
            if(next_location != location): 
                legal_action[i] = 1
                next_distance[i] = self.getOpDistance(next_location, op_location)
                legal_action_total += 1
            #else: next_distance[i] = now_distance
        # 求每个动作的概率
        near_goal_total = 0.0
        for i in range(self.action_num):
            # 先将离对手近的动作的值变大，从而求得更大的概率
            if legal_action[i] != 0:
                next_distance[i] = 1.0 / next_distance[i]
                near_goal_total += next_distance[i]
        # 求离对手近的动作概率
        for i in range(self.action_num):
            if legal_action[i] != 0: action_maxtrix[i] = (1 - define.action_rate) * (1 / legal_action_total) + define.action_rate * next_distance[i] / near_goal_total
        
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        max_dis = np.max(next_distance)
        while(True):
            rand = random.randint(0, 3)
            if legal_action[rand] != 0 and max_dis == next_distance[rand]:
                action_maxtrix[rand] = 1
                break
        #print next_distance
        return action_maxtrix

    """
    不持球靠近自己球门

    """
    def doPolicy3(self, player_state, location):
        # 4个动作，0代表左 1代表上 2代表右 3代表下
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        legal_action = np.zeros(self.action_num, dtype=float)
        next_distance = np.zeros(self.action_num, dtype=float)
        # 因为是以自家球门为目标
        now_distance = self.getGoalDistance(location, 1 - player_state)
        # 符合要求的动作个数
        legal_action_total = 0.0
        # 先找出合格的动作
        for i in range(self.action_num):
            next_location = self.getNextPosition(i, player_state, location)
            if(next_location != location): 
                legal_action[i] = 1
                next_distance[i] = self.getGoalDistance(next_location, 1 - player_state)
                legal_action_total += 1
            #else: next_distance[i] = now_distance
        # 求每个动作的概率 离自家球门越近动作概率越大
        near_goal_total = 0.0
        for i in range(self.action_num):
            if legal_action[i] != 0:
                next_distance[i] = 1.0 / next_distance[i]
                near_goal_total += next_distance[i]
        for i in range(self.action_num):
            if legal_action[i] != 0: action_maxtrix[i] = (1 - define.action_rate) * (1 / legal_action_total) + define.action_rate * (1 / next_distance[i]) / near_goal_total
        
        action_maxtrix = np.zeros(self.action_num, dtype=float)
        max_dis = np.max(next_distance)
        while(True):
            rand = random.randint(0, 3)
            if legal_action[rand] != 0 and max_dis == next_distance[rand]:
                action_maxtrix[rand] = 1
                break
        #print next_distance
        return action_maxtrix

    def getNextPosition(self, action, play_state, location):
        location1 = [0, 0]
        location1[0] = location[0]
        location1[1] = location[1]
        # 球员A
        if action == self.left:
            if(self.IsLegalAction(play_state, [location[0] - 1, location[1]])):
                location1[0] = location1[0] - 1
        elif action == self.right:
            if(self.IsLegalAction(play_state, [location[0] + 1, location[1]])):
                location1[0] = location1[0] + 1
        elif action == self.up:
            if(self.IsLegalAction(play_state, [location[0], location[1] - 1])):
                location1[1] = location1[1] - 1
        elif action == self.down:
            if(self.IsLegalAction(play_state, [location[0], location[1] + 1])):
                location1[1] = location1[1] + 1
        return location1

    def IsLegalAction(self, player_state, position):
        # 球门区域
        if(player_state == 0 and (position in self.ball_game.goal_B)):
            return True
        elif(player_state == 1 and (position in self.ball_game.goal_A)):
            return True
        # 球门以外的区域
        if(position[0] < 1 or position[0] >= self.ball_game.game_width - 1):
            return False
        if(position[1] < 0 or position[1] >= self.ball_game.game_height):
            return False
        return True

    # 以点为目标的距离
    def getOpDistance(self, location, op_location):
        return (math.fabs(location[0] - op_location[0]) + math.fabs(location[1] - op_location[1])) + 1

    # 以球门为目标的距离吗
    def getGoalDistance(self, location, player_state):
        dis = []
        goal_locations = []
        if player_state == 0:
            goal_locations = self.ball_game.goal_B
        elif player_state == 1:
            goal_locations = self.ball_game.goal_A
        for g in goal_locations:
            dis.append(self.getOpDistance(location, g))
        return np.min(dis)

