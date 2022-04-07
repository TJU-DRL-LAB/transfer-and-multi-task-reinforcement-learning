# coding=utf-8
import numpy as np
import imageio
from gym import spaces
import tkinter as tk
from PIL import Image, ImageTk 
import matplotlib.pyplot  as plt
import time

CELL, BLOCK, AGENT_GOAL, OPPONENT_GOAL, AGENT, OPPONENT = range(6)
WIN, LOSE = 5, -5
UP, RIGHT, DOWN, LEFT, HOLD = range(5)

UNIT = 40


class Soccer(tk.Tk, object):
    playground = [1, 1, 1, 1, 1, 1, 1,
                  1, 0, 0, 0, 0, 0, 1,
                  3, 0, 0, 0, 0, 0, 2,
                  3, 0, 0, 0, 0, 0, 2,
                  3, 0, 0, 0, 0, 0, 2,
                  1, 0, 0, 0, 0, 0, 1,
                  1, 1, 1, 1, 1, 1, 1]
    action_map = {
        UP: np.array([-1, 0]),
        RIGHT: np.array([0, 1]),
        DOWN: np.array([1, 0]),
        LEFT: np.array([0, -1]),
        HOLD: np.array([0, 0])}

    def __init__(self):
        super(Soccer, self).__init__()
        self.size = 7
        self.agent = np.array([3, 1])
        self.opponent = np.array([3, 5])
        self.grids = np.array(self.playground).reshape(self.size, self.size)
        self.agent_keep_ball = False
        self.action_space = [UP, RIGHT, DOWN, LEFT, HOLD]  
        self.n_actions = len(self.action_space)
        self.n_features = 5
        self.visualize()
        # low high to observe
        #self.observation_space = spaces.Discrete(7 * 7 * 2)

    def step(self, act_a, act_o):
        new_pos_a = self.agent + self.action_map[act_a]
        new_pos_o = self.opponent + self.action_map[act_o]

        reward, done, s_ = 0, False, []
        # opponent win
        if self.grids[tuple(new_pos_o)] == 3 and not self.agent_keep_ball:
            reward = LOSE
            done = True

        # agent win
        if self.grids[tuple(new_pos_a)] == 2 and self.agent_keep_ball:
            reward = WIN
            done = True

        # valid check for opponent and agent
        if self.grids[tuple(new_pos_a)] in (1, 2, 3):
            new_pos_a = self.agent

        if self.grids[tuple(new_pos_o)] in (1, 2, 3):
            new_pos_o = self.opponent

        # collision
        if np.array_equal(new_pos_a, new_pos_o) and self.grids[tuple(new_pos_a)] != 1:
            self.agent_keep_ball = not self.agent_keep_ball
        
        #print(self.canvas.coords(self.agent_rect))
        self.agent = new_pos_a
        self.opponent = new_pos_o

        self.canvas.delete(self.agent_rect)
        self.canvas.delete(self.opp_rect)
        self.agent_rect = self.canvas.create_rectangle(self.agent[1] * UNIT, self.agent[0] * UNIT, (self.agent[1] + 1) * UNIT, (self.agent[0] + 1) * UNIT, fill='red')
        self.opp_rect = self.canvas.create_rectangle(self.opponent[1] * UNIT, self.opponent[0] * UNIT, (self.opponent[1] + 1) * UNIT, (self.opponent[0] + 1) * UNIT, fill='blue')
        
        self.canvas.delete(self.ball_rect)
        if self.agent_keep_ball:
            self.ball_rect = self.canvas.create_oval((self.agent[1] * UNIT, self.agent[0] * UNIT, (self.agent[1] + 1) * UNIT, (self.agent[0] + 1) * UNIT), fill='white')
        else:
            self.ball_rect = self.canvas.create_oval(self.opponent[1] * UNIT, self.opponent[0] * UNIT, (self.opponent[1] + 1) * UNIT, (self.opponent[0] + 1) * UNIT, fill='white')

        s_ = [self.agent[0], self.agent[1], self.opponent[0], self.opponent[1]]
        if self.agent_keep_ball:
            s_.append(0)
        else: s_.append(1)
        s_ = np.array(s_[:5])/ 10
        return s_, reward, done

    # reset position and ball
    def reset(self):
        self.agent = np.array([3, 1])
        self.opponent = np.array([3, 5])
        self.agent_keep_ball = False
        self.update()
        s_ = [self.agent[0], self.agent[1], self.opponent[0], self.opponent[1]]
        if self.agent_keep_ball:
            s_.append(0)
        else: s_.append(1)
        s_ = np.array(s_[:5])/ 10
        return s_

    # render array
    def render(self):
        m = np.copy(self.grids)
        m[tuple(self.agent)] = 4
        m[tuple(self.opponent)] = 5
        if self.agent_keep_ball:
            m[tuple(self.agent)] += 2
        else:
            m[tuple(self.opponent)] += 2
        #print(m, end='\n\n')
        self.update()
        return m.reshape(49)

    # render img
    def visualize(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=self.size * UNIT,
                           width=self.size * UNIT)
        # create grids
        for c in range(0, self.size * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, self.size * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.size * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, self.size * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        m = np.copy(self.grids)
        m[tuple(self.agent)] = 4
        m[tuple(self.opponent)] = 5
        #print(m)
        for j in range(self.size):
            for i in range(self.size):
                if m[j, i] == 1: self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='black')
                elif m[j, i] == 2 or m[j, i] == 3: self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='white')
                elif m[j, i] == 0 or m[j, i] == 4 or m[j, i] == 5: self.canvas.create_rectangle(i * UNIT, j * UNIT, (i + 1) * UNIT, (j + 1) * UNIT, fill='green')
        
        self.agent_rect = self.canvas.create_rectangle(self.agent[1] * UNIT, self.agent[0] * UNIT, (self.agent[1] + 1) * UNIT, (self.agent[0] + 1) * UNIT, fill='red')
        self.opp_rect = self.canvas.create_rectangle(self.opponent[1] * UNIT, self.opponent[0] * UNIT, (self.opponent[1] + 1) * UNIT, (self.opponent[0] + 1) * UNIT, fill='blue')
        
        if self.agent_keep_ball:
            self.ball_rect = self.canvas.create_oval((self.agent[0] * UNIT, self.agent[0] * UNIT, (self.agent[1] + 1) * UNIT, (self.agent[1] + 1) * UNIT), fill='white')
        else:
            self.ball_rect = self.canvas.create_oval(self.opponent[1] * UNIT, self.opponent[0] * UNIT, (self.opponent[1] + 1) * UNIT, (self.opponent[0] + 1) * UNIT, fill='white')

        # pack all
        self.canvas.pack()


if __name__ == '__main__':
    env = Soccer()
    env.reset()
    # agent strategy
    agent_actions = [RIGHT, RIGHT, UP, RIGHT, RIGHT, RIGHT]
    # opponent strategy, you can initialize it randomly
    opponent_actions = [UP, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT]

    for a_a, a_o in zip(agent_actions, opponent_actions):
        env.render() 
        env.step(a_a, a_o)
        time.sleep(1)
        #env.after(100, run_maze)
        #env.mainloop()
        # env.render()
