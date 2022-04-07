from games import Soccer
from RL_brain import DeepQNetwork
import matplotlib.pyplot as plt
from Players import Player
import numpy as np

UP, RIGHT, DOWN, LEFT, HOLD = range(5)

opponent_actions_list = [[UP, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT],
                    [DOWN, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT],
                    [UP, UP, LEFT, LEFT, LEFT, LEFT, DOWN, LEFT],
                    [DOWN, DOWN, LEFT, LEFT, LEFT, LEFT, UP, LEFT]]

path_list = ["policy1", "policy2", "policy3", "policy4"]

step_len_list = [6, 6, 7, 7]

env = Soccer()

RL1 = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
RL2 = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )

RL3 = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
RL4 = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=1,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )


RL1.load_model(path_list[0])
RL2.load_model(path_list[1])
RL3.load_model(path_list[2])
RL4.load_model(path_list[3])
RL_list = [RL1, RL2, RL3, RL4]

player_A = Player(0)
player_B = Player(1)


def run(RL, step_len, opponent_actions):
    observation = env.reset()
    step = 0
    while True:
        #env.render()
        action = RL.choose_action(observation)
        if step > step_len:
            opp_action = HOLD
        else: opp_action = opponent_actions[step]
        observation_, reward, done = env.step(action, opp_action)
        observation = observation_
        if done or step > 50:
            return reward
        step += 1


def get_reward(policyA, policyB):
    reward = run(RL_list[policyA], step_len_list[policyB], opponent_actions_list[policyB])
    reward /= 5
    return [reward, - reward]


def initPerformanceModel():
    # player与对手交互，每个策略玩10次
    for i in range(4):
        for j in range(4):
            reward_total = [0, 0]
            for m in range(1000):
                reward = get_reward(i, j)
                reward_total[0] += reward[0]
                reward_total[1] += reward[1]
                player_A.updatePerformanceModel(i, j, reward[0], reward[1])
                player_B.updatePerformanceModel(j, i, reward[1], reward[0])
            print(i, j)
            print(reward_total)


def interaction():
    # 初始化performance_model
    #initPerformanceModel()
    #player_A.savePerformanceModel()
    #player_B.savePerformanceModel()
    # load performance model
    player_A.loadPerformanceModel()
    player_B.loadPerformanceModel()
    player_A.tom.reset()
    player_B.tom.reset()
    # 交互600次
    reward_list = []
    reward_total = [0, 0]

    #policy_B = random.randint(0, 3)
    for i in range(200):
        # 选择策略
        policy_A = player_A.choosePolicy()
        policy_B = player_B.choosePolicy()

        # 玩游戏
        reward = get_reward(policy_A, policy_B)

        # 更新参数
        player_A.updatePolicy(policy_A, policy_B, reward[0], reward[1])
        player_B.updatePolicy(policy_B, policy_A, reward[1], reward[0])
        reward_total[0] += reward[0]
        reward_total[1] += reward[1]
        temp = [0, 0]
        temp[0] = reward_total[0]
        temp[1] = reward_total[1]
        reward_list.append(reward[0])

    #policy_B = random.randint(0, 3)
    for i in range(200):
        # 选择策略
        policy_A = player_A.choosePolicy()
        policy_B = player_B.choosePolicy()

        # 玩游戏
        reward = get_reward(policy_A, policy_B)

        # 更新参数
        player_A.updatePolicy(policy_A, policy_B, reward[0], reward[1])
        player_B.updatePolicy(policy_B, policy_A, reward[1], reward[0])
        reward_total[0] += reward[0]
        reward_total[1] += reward[1]
        temp = [0, 0]
        temp[0] = reward_total[0]
        temp[1] = reward_total[1]
        reward_list.append(reward[0])

    #policy_B = random.randint(0, 3)
    for i in range(200):
        # 选择策略
        policy_A = player_A.choosePolicy()
        policy_B = player_B.choosePolicy()

        # 玩游戏
        reward = get_reward(policy_A, policy_B)

        # 更新参数
        player_A.updatePolicy(policy_A, policy_B, reward[0], reward[1])
        player_B.updatePolicy(policy_B, policy_A, reward[1], reward[0])
        reward_total[0] += reward[0]
        reward_total[1] += reward[1]
        temp = [0, 0]
        temp[0] = reward_total[0]
        temp[1] = reward_total[1]
        reward_list.append(reward[0])
    print(reward_total)
    return reward_list


if __name__ == "__main__":
    reward_list_100 = np.zeros(600, dtype=float)
    for i in range(20):
        reward_list = interaction()
        reward_list = np.array(reward_list)
        reward_list_100 = reward_list_100 + reward_list
    reward_list_100 = reward_list_100[:600] / 20
    
    plt.plot(np.arange(len(reward_list_100)), reward_list_100)
    plt.ylabel('Cost')
    plt.xlabel('training steps')
    plt.show()
    np.save('tom1_static_change_20_score.npy', reward_list_100)
    print(reward_list_100.reshape(600, 1))
    #env.mainloop()
