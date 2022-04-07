# -*- coding:utf-8 -*- 
import numpy as np
import math
from define import define
import random


class PolicyTF():
    def __init__(self):
        # 策略0， 去目标点1 -> 2 -> 3 ->4
        # 策略1， 去目标点1 -> 2 -> 4 ->3
        # 策略2， 去目标点1 -> 3 -> 2 ->4
        # 策略3， 去目标点1 -> 3 -> 4 ->2
        # 策略4， 去目标点1 -> 4 -> 3 ->2
        # 策略5， 去目标点1 -> 4 -> 2 ->3
        # 策略6， 去目标点2 -> 1 -> 3 ->4
        # 策略7， 去目标点2 -> 1 -> 4 ->3
        # 策略8， 去目标点2 -> 3 -> 1 ->4
        # 策略9， 去目标点2 -> 3 -> 4 ->1
        # 策略10， 去目标点2 -> 4 -> 1 ->3
        # 策略11， 去目标点2 -> 4 -> 3 ->1
        # 策略12， 去目标点3 -> 1 -> 2 ->4
        # 策略13， 去目标点3 -> 1 -> 4 ->2
        # 策略14， 去目标点3 -> 2 -> 1 ->4
        # 策略15， 去目标点3 -> 2 -> 4 ->1
        # 策略16， 去目标点3 -> 4 -> 1 ->2
        # 策略17， 去目标点3 -> 4 -> 2 ->1
        # 策略18， 去目标点4 -> 1 -> 3 ->2
        # 策略19， 去目标点4 -> 1 -> 2 ->3
        # 策略20， 去目标点4 -> 2 -> 1 ->3
        # 策略21， 去目标点4 -> 2 -> 3 ->1
        # 策略22， 去目标点4 -> 3 -> 1 ->2
        # 策略23， 去目标点4 -> 3 -> 2 ->1
        self.policy = define.policy_num
        self.policyList = []
        self.init()
        self.action_num = 5
        # 动作
        self.left = 0
        self.up = 1
        self.right = 2
        self.down = 3
        self.stay = 4

    def init(self):
        self.policyList.append([1, 2, 3, 4])
        self.policyList.append([1, 2, 4, 3])
        self.policyList.append([1, 3, 2, 4])
        self.policyList.append([1, 3, 4, 2])
        self.policyList.append([1, 4, 3, 2])
        self.policyList.append([1, 4, 2, 3])
        self.policyList.append([2, 1, 3, 4])
        self.policyList.append([2, 1, 4, 3])
        self.policyList.append([2, 3, 1, 4])
        self.policyList.append([2, 3, 4, 1])
        self.policyList.append([2, 4, 1, 3])
        self.policyList.append([2, 4, 3, 1])
        self.policyList.append([3, 1, 2, 4])
        self.policyList.append([3, 1, 4, 2])
        self.policyList.append([3, 2, 1, 4])
        self.policyList.append([3, 2, 4, 1])
        self.policyList.append([3, 4, 1, 2])
        self.policyList.append([3, 4, 2, 1])
        self.policyList.append([4, 1, 3, 2])
        self.policyList.append([4, 1, 2, 3])
        self.policyList.append([4, 2, 1, 3])
        self.policyList.append([4, 2, 3, 1])
        self.policyList.append([4, 3, 1, 2])
        self.policyList.append([4, 3, 2, 1])

    # policy_index: 策略类型  目标点
    def getGoals(self, policy_index, catch_goals_num):
        try:
            return self.policyList[policy_index][catch_goals_num]
        except Exception as err:
            print(policy_index, catch_goals_num)
