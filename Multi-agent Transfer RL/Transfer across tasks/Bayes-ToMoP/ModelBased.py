# -*- coding:utf-8 -*-
from BallGame import BallGame
from State import State
from define import define
import numpy


class ModelBased(object):
    def __init__(self):
        self.ball_game = BallGame()
        self.state_list = []
        self.delta = 0
        self.cur_row = 0
        self.cur_column = 0
        self.cur_action = 0
        self.action_num = 4
        self.has_ball = 0
        self.play_state = 0
        self.cur_state_list = []
        self.greedy = define.greedy
        self.learning = define.learning
        for i in range(self.ball_game.game_width):
            for j in range(self.ball_game.game_height):
                state = State(i, j)
                self.state_list.append(state)
                #print i, j, state.T
        

    def getAction(self, position, player_state, has_ball):
        self.cur_row = position[0]
        self.cur_column = position[1]
        self.has_ball = has_ball
        self.play_state = player_state
        state_index = position[0] * self.ball_game.game_height + position[1]
        #action = self.state_list[state_index].chooseAction(player_state, has_ball, self.greedy)
        action = self.state_list[state_index].chooseActionQ(player_state, has_ball, self.greedy)
        self.cur_action = action
        return action

    def update(self, reward):
        state_index = self.cur_row * self.ball_game.game_height + self.cur_column
        self.state_list[state_index].updateReward(reward, self.cur_action)
        if ([self.cur_row, self.cur_column, self.cur_action] in self.cur_state_list) == False and self.state_list[state_index].n[self.cur_action] > define.m:
            self.cur_state_list.append([self.cur_row, self.cur_column, self.cur_action]) 
        for m in self.cur_state_list:
            #if m in self.ball_game.goal_A or m in self.ball_game.goal_B:
            #    continue
            V_list = numpy.zeros(self.action_num, dtype=float)
            state_index_temp = m[0] * self.ball_game.game_height + m[1]
            next_state = self.state_list[state_index_temp].getNextState(m[2])
            next_state_index = next_state[0] * self.ball_game.game_height + next_state[1]
            if self.has_ball == 0:
                self.state_list[state_index_temp].Q[m[2]] = self.state_list[state_index_temp].getR(m[2]) + define.discount * self.state_list[next_state_index].getMaxQ(self.has_ball)
            else:
                self.state_list[state_index_temp].Q1[m[2]] = self.state_list[state_index_temp].getR(m[2]) + define.discount * self.state_list[next_state_index].getMaxQ(self.has_ball)

    def update1(self, reward):
        state_index = self.cur_row * self.ball_game.game_height + self.cur_column
        self.state_list[state_index].updateReward(reward, self.cur_action)
        self.cur_state_list.append(state_index)
        #print [self.cur_row, self.cur_column]

    def clear(self):
        #self.cur_state_list = []
        self.greedy = self.greedy * (1 - define.greedy_discount)
        self.learning = self.learning * (1 - define.learning_discount)

    def updateState(self):
        index = len(self.cur_state_list) - 1
        result = []
        while index >= 0:
            V_list = numpy.zeros(self.action_num, dtype=float)
            for i in range(self.action_num):
                next_state = self.state_list[self.cur_state_list[index]].getNextState(i)
                if not self.IsLegalAction(next_state):
                    V_list[i] = -100
                else:
                    next_state_index = next_state[0] * self.ball_game.game_height + next_state[1]
                    #print next_state
                    V_list[i] = self.state_list[self.cur_state_list[index]].getR(i) + define.discount * self.state_list[next_state_index].getV(self.has_ball)
            self.state_list[self.cur_state_list[index]].setV(self.has_ball, max(V_list))
            self.state_list[self.cur_state_list[index]].setMaxAction(self.has_ball, numpy.argmax(V_list))
            result.append([self.cur_state_list[index], max(V_list),  numpy.argmax(V_list)])
            #print V_list, max(V_list), numpy.argmax(V_list)
            index -= 1
        #print(result)

    def updateQ(self, reward):
        state_index = self.cur_row * self.ball_game.game_height + self.cur_column
        self.state_list[state_index].updateReward(reward, self.cur_action)
        next_state = self.state_list[state_index].getNextState(self.cur_action)
        next_state_index = next_state[0] * self.ball_game.game_height + next_state[1]
        if self.has_ball == 0:
            q = self.state_list[state_index].Q[self.cur_action]
            q = q + self.learning * (reward - q + define.discount * self.state_list[next_state_index].getMaxQ(self.has_ball))
            
            self.state_list[state_index].Q[self.cur_action] = q
        else:
            q = self.state_list[state_index].Q1[self.cur_action]
            temp = q
            q = q + self.learning * (reward - q + define.discount * self.state_list[next_state_index].getMaxQ(self.has_ball))
            if q > 1:
                print(temp, self.learning ,reward, self.state_list[next_state_index].getMaxQ(self.has_ball))
            self.state_list[state_index].Q1[self.cur_action] = q
        #state = [state_index, self.cur_action, reward]
        #self.cur_state_list.append(state)

    def updateQState(self):
        index = 0
        a = []
        while index < len(self.cur_state_list):
            state_index = self.cur_state_list[index][0]
            action = self.cur_state_list[index][1]
            reward = self.cur_state_list[index][2]
            next_state = self.state_list[state_index].getNextState(action)
            next_state_index = next_state[0] * self.ball_game.game_height + next_state[1]
            if self.has_ball == 0:
                q = self.state_list[state_index].Q[action]
                q = q + self.state_list[state_index].learning * (reward - q + define.discount * self.state_list[next_state_index].getMaxQ(self.has_ball))
                self.state_list[state_index].Q[action] = q
                a.append([state_index, q])
            else:
                q = self.state_list[state_index].Q1[action]
                q = q + self.state_list[state_index].learning * (reward - q + define.discount * self.state_list[next_state_index].getMaxQ(self.has_ball))
                self.state_list[state_index].Q1[action] = q
                #a.append([state_index, q])
            index += 1
        #print (a)

    def IsLegalAction(self, position):
        # 球门区域
        if self.play_state == 0 and (position in self.ball_game.goal_B):
            return True
        elif self.play_state == 1 and (position in self.ball_game.goal_A):
            return True
        # 球门以外的区域
        if position[0] < 1 or position[0] >= self.ball_game.game_width - 1:
            return False
        if position[1] < 0 or position[1] >= self.ball_game.game_height:
            return False
        return True

    def printStateList(self):
        for i in range(self.ball_game.game_width):
            for j in range(self.ball_game.game_height):
                index = i * self.ball_game.game_height + j
                #print (i, j, self.state_list[index].Q)
        #start = [1, 1]
        #state = []
        #state.append([start[0], start[1]])
        #while(len(state) < 10):
        #    state_index = start[0] * self.ball_game.game_height + start[1]
        #    action = self.state_list[state_index].maxAction[0]
        #    next_state = self.state_list[state_index].T[action][1]
        #    start = next_state
        #    state.append([start[0], start[1]])
        #print state
        start = [1, 1]
        state = []
        state.append([start[0], start[1]])
        while len(state) < 10:
            state_index = start[0] * self.ball_game.game_height + start[1]
            action = self.state_list[state_index].chooseActionQ(self.play_state, self.has_ball, 0)
            next_state = self.state_list[state_index].T[action][1]
            start = next_state
            state.append([start[0], start[1]])
        #print (state)
        

