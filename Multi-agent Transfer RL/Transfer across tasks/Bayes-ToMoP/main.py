from Play import Play
import numpy
if __name__ == "__main__":
    reward_list_100 = []
    for i in range(50):
        p = Play()
        reward_list = p.interaction()
        reward_list_100.append(reward_list)
    print(reward_list_100)
    numpy.save('tom1_tom0_50_score.npy', reward_list_100)

