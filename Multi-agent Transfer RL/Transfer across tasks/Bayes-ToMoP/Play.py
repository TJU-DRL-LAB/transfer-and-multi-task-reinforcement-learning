# -*- coding:utf-8 -*- 
from Players import Player
from Thieves_and_hunters import Thieves_and_hunters
import random


class Play(object):
    def __init__(self):
        # left: 0 right: 1
        self.player_A = Player(0)
        self.player_B = Player(1)
        self.Thieves_and_hunters = Thieves_and_hunters()

    def interaction(self):
        # 初始化performance_model
        #self.initPerformanceModel()
        #self.player_A.savePerformanceModel()
        #self.player_B.savePerformanceModel()
        # Load performance_model
        self.player_A.loadPerformanceModel()
        self.player_B.loadPerformanceModel()
        self.player_A.tom.reset()
        self.player_B.tom.reset()
        #print(self.player_B.tom.performance_model.performance_model)
        #print(self.player_A.tom.performance_model.performance_model)

        # 交互1000次
        reward_list = []
        reward_total = [0, 0]
        for i in range(200):
            # 选择策略
            policy_A = self.player_A.choosePolicy()
            policy_B = self.player_B.choosePolicy()

            # 玩游戏
            reward = self.playGame(policy_A, policy_B, i)
            # 更新参数
            self.player_A.updatePolicy(policy_A, policy_B, reward[0], reward[1])
            self.player_B.updatePolicy(policy_B, policy_A, reward[1], reward[0])
            reward_total[0] += reward[0]
            reward_total[1] += reward[1]
            temp = [0, 0]
            temp[0] = reward_total[0]
            temp[1] = reward_total[1]
            reward_list.append(temp)
        for i in range(200):
            # 选择策略
            policy_A = self.player_A.choosePolicy()
            policy_B = self.player_B.choosePolicy()

            # 玩游戏
            reward = self.playGame(policy_A, policy_B, i)
            # 更新参数
            self.player_A.updatePolicy(policy_A, policy_B, reward[0], reward[1])
            self.player_B.updatePolicy(policy_B, policy_A, reward[1], reward[0])
            reward_total[0] += reward[0]
            reward_total[1] += reward[1]
            temp = [0, 0]
            temp[0] = reward_total[0]
            temp[1] = reward_total[1]
            reward_list.append(temp)
        for i in range(200):
            # 选择策略
            policy_A = self.player_A.choosePolicy()
            policy_B = 2#self.player_B.choosePolicy()

            # 玩游戏
            reward = self.playGame(policy_A, policy_B, i)
            # 更新参数
            self.player_A.updatePolicy(policy_A, policy_B, reward[0], reward[1])
            self.player_B.updatePolicy(policy_B, policy_A, reward[1], reward[0])
            reward_total[0] += reward[0]
            reward_total[1] += reward[1]
            temp = [0, 0]
            temp[0] = reward_total[0]
            temp[1] = reward_total[1]
            reward_list.append(temp)
            #print reward
        print(reward_total)
        return reward_list

    def initPerformanceModel(self):
        # player与对手交互，每个策略玩10次
        for i in range(self.player_A.getPolicyNum()):
            for j in range(self.player_B.getPolicyNum()):
                reward_total = [0, 0]
                for m in range(1000):
                    reward = self.playGame(i, j)
                    reward_total[0] += reward[0]
                    reward_total[1] += reward[1]
                    #print reward
                    self.player_A.updatePerformanceModel(i, j, reward[0], reward[1])
                    self.player_B.updatePerformanceModel(j, i, reward[1], reward[0])
                print(i, j)
                print(reward_total)

    # tabular soccer
    def playGame(self, policy_A, policy_B,t):
        # 设置策略
        self.player_A.setPolicyIndex(policy_A)
        self.player_B.setPolicyIndex(policy_B)
        # 设置球权归属 0表示有球，1表示无球 这里给player_A定球权，player_B就是相反的球权
        rand = random.randint(0, 1)
        self.player_A.setRightToServe(rand)
        self.player_B.setRightToServe(1 - rand)
        # 定义初始位置
        self.player_A.setLocation()
        self.player_B.setLocation()
        self.player_A.setOpLocation(self.player_B.getLocation())
        self.player_B.setOpLocation(self.player_A.getLocation())
        # 开始 50次如果还没得分自动结束，双方都为0分
        for i in range(50):
            #print self.player_A.getLocation()
            #print self.player_B.getLocation()
            action_A = self.player_A.chooseAction()
            action_B = self.player_B.chooseAction()
            # 选手移动,顺序随机
            rand = random.randint(0, 1)
            if rand == 0: #A先走
                self.player_A.move(action_A)
                # 判断A是否得分
                if self.player_A.IsScored():
                    return [1, 0]
                # 判断是否交换球权
                if self.player_A.getLocation() == self.player_B.getLocation():
                    self.player_A.setRightToServe(1 - self.player_A.getBallRight())
                    self.player_B.setRightToServe(1 - self.player_B.getBallRight())                
                self.player_B.move(action_B)
                # 判断B是否得分
                if self.player_B.IsScored():
                    return [0, 1]
                # 判断是否交换球权
                if self.player_A.getLocation() == self.player_B.getLocation():
                    self.player_A.setRightToServe(1 - self.player_A.getBallRight())
                    self.player_B.setRightToServe(1 - self.player_B.getBallRight())
            else: #B先走
                self.player_B.move(action_B)
                # 判断B是否得分
                if self.player_B.IsScored():
                    return [0, 1]
                # 判断是否交换球权
                if self.player_A.getLocation() == self.player_B.getLocation():
                    self.player_A.setRightToServe(1 - self.player_A.getBallRight())
                    self.player_B.setRightToServe(1 - self.player_B.getBallRight())                
                self.player_A.move(action_A)
                # 判断A是否得分
                if self.player_A.IsScored():
                    return [1, 0]
                # 判断是否交换球权
                if self.player_A.getLocation() == self.player_B.getLocation():
                    self.player_A.setRightToServe(1 - self.player_A.getBallRight())
                    self.player_B.setRightToServe(1 - self.player_B.getBallRight())           
            # 获取对手位置
            self.player_A.setOpLocation(self.player_B.getLocation())
            self.player_B.setOpLocation(self.player_A.getLocation())
           
            #print self.player_A.getBallRight(), self.player_B.getBallRight()
        # 50次还没分出胜负的
        return [0, 0]
    '''
    #thieves and hunters playGame
    def playGame(self, policy_A, policy_B,t):
        # 设置策略
        self.player_A.setPolicyIndex(policy_A)
        self.player_B.setPolicyIndex(policy_B)
        self.Thieves_and_hunters.init()
        catch_goals_num_A = 0
        catch_goals_num_B = 0
        # 开始 50次如果还没得分自动结束，双方都为0分
        for i in range(50):
            catch_goals_num_A = self.Thieves_and_hunters.catch_goals_num_A
            catch_goals_num_B = self.Thieves_and_hunters.catch_goals_num_B
            goals_index_A = self.player_A.getGoalsIndex(catch_goals_num_A)
            action_A = self.Thieves_and_hunters.chooseAction(goals_index_A, 'A')
            if catch_goals_num_B < len(self.Thieves_and_hunters.goals):
                # print('a')
                goals_index_B = self.player_B.getGoalsIndex(catch_goals_num_B)
                action_B = self.Thieves_and_hunters.chooseAction(goals_index_B, 'B')
            else:
                action_B = self.player_B.stay
            self.Thieves_and_hunters.move(action_A, 'A')
            self.Thieves_and_hunters.move(action_B, 'B')
            self.Thieves_and_hunters.check()
            if self.Thieves_and_hunters.endGame():
                break

        return [self.Thieves_and_hunters.A_score, self.Thieves_and_hunters.B_score]

    # Model_based learning
    
    def playGame(self, policy_A, policy_B, t):
        # 设置策略
        self.player_A.setPolicyIndex(policy_A)
        self.player_B.setPolicyIndex(policy_B)
        # 设置球权归属 0表示有球，1表示无球 这里给player_A定球权，player_B就是相反的球权
        rand = random.randint(0, 1)
        self.player_A.setRightToServe(1)
        self.player_B.setRightToServe(0)
        self.player_A.has_ball_cur = 1
        # 定义初始位置
        self.player_A.setLocation()
        self.player_B.setLocation()
        self.player_A.setOpLocation(self.player_B.getLocation())
        self.player_B.setOpLocation(self.player_A.getLocation())
        if t > 6000:
            self.player_A.printStateList()
        self.player_A.Model_Based.clear()
        # print t,self.player_A.Model_Based.greedy
        # 开始 50次如果还没得分自动结束，双方都为0分
        state = []
        for i in range(200):
            action_A = self.player_A.chooseAction()
            action_B = self.player_B.chooseAction()
            # print self.player_A.getLocation(),self.player_B.getLocation(), action_A, action_B
            location = self.player_A.getLocation ()
            state.append([location[0], location[1], self.player_A.has_ball_cur])
            # 选手移动,顺序随机
            rand = random.randint(0, 1)
            temp = 0
            if rand == 0:  # A先走
                self.player_A.move(action_A)
                # 判断A是否得分
                if self.player_A.IsScored():
                    # self.player_A.Model_Based.updateQState()
                    self.player_A.Model_Based.update(2)
                    if t > 5900:
                        print(state)
                    return [1, 0]
                    # 判断是否交换球权
                if self.player_A.getLocation() == self.player_B.getLocation() and temp == 0:
                    self.player_A.setRightToServe(1 - self.player_A.getBallRight())
                    self.player_B.setRightToServe(1 - self.player_B.getBallRight())
                    if self.player_A.getBallRight() == 0:
                        self.player_A.Model_Based.update(1)
                    else:
                        self.player_A.Model_Based.update(-1)
                else:
                    temp += 1
                self.player_B.move(action_B)
                # 判断B是否得分
                if self.player_B.IsScored():
                    self.player_A.Model_Based.update(0)
                    # self.player_A.Model_Based.updateQState()
                    if t > 5900:
                        print(state)
                    return [0, 1]
                # 判断是否交换球权
                if self.player_A.getLocation() == self.player_B.getLocation() and temp == 1:
                    self.player_A.setRightToServe(1 - self.player_A.getBallRight())
                    self.player_B.setRightToServe(1 - self.player_B.getBallRight())
                    if self.player_A.getBallRight() == 0:
                        self.player_A.Model_Based.update(1)
                    else:
                        self.player_A.Model_Based.update(-1)
                else:
                    temp += 1
                if temp == 2:
                    self.player_A.Model_Based.update(0)
            else:  # B先走
                self.player_B.move(action_B)
                # 判断B是否得分
                if self.player_B.IsScored():
                    self.player_A.Model_Based.update(0)
                    # self.player_A.Model_Based.updateQState()
                    if t > 5900:
                        print(state)
                    return [0, 1]
                # 判断是否交换球权
                if self.player_A.getLocation() == self.player_B.getLocation() and temp == 0:
                    self.player_A.setRightToServe(1 - self.player_A.getBallRight())
                    self.player_B.setRightToServe(1 - self.player_B.getBallRight())
                    if self.player_A.getBallRight() == 0:
                        self.player_A.Model_Based.update(1)
                    else:
                        self.player_A.Model_Based.update (-1)
                else:
                    temp += 1
                self.player_A.move(action_A)
                # 判断A是否得分
                if self.player_A.IsScored():
                    self.player_A.Model_Based.update(2)
                    # self.player_A.Model_Based.updateQState()
                    if t > 5900:
                        print (state)
                    return [1, 0]
                # 判断是否交换球权
                if self.player_A.getLocation() == self.player_B.getLocation() and temp == 1:
                    self.player_A.setRightToServe(1 - self.player_A.getBallRight())
                    self.player_B.setRightToServe(1 - self.player_B.getBallRight())
                    if self.player_A.getBallRight() == 0:
                        self.player_A.Model_Based.update(1)
                    else:
                        self.player_A.Model_Based.update(-1)
                else:
                    temp += 1
                if temp == 2:
                    self.player_A.Model_Based.update(0)
            # 获取对手位置
            self.player_A.setOpLocation(self.player_B.getLocation())
            self.player_B.setOpLocation(self.player_A.getLocation())

            # print self.player_A.getBallRight(), self.player_B.getBallRight()
        # 50次还没分出胜负的
        if t > 5900:
            print(state)
        self.player_A.Model_Based.update(0)
        return [0, 0]
        '''

