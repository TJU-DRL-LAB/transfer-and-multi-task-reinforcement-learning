# 5m_vs_6m 6m 7m 8m_vs_9m 8m_vs_10m 10m 15m 20m 25m     7m_vs_9m
# target_env = {'7m_vs_9m':[7,9]}
# envs = {
#     '5m_vs_6m':[5,6],
#     '6m':[6,6],
#     '7m':[7,7],
#     '8m_vs_9m':[8,9],
#     '8m_vs_10m':[8,10],
#     '10m':[10,10],
#     '15m':[15,15],
#     '20m':[20,20],
#     '25m':[25,25]
# }

# 2s4z 2s5z 3s4z 2s3z_vs_2s4z 2s3z_vs_2s5z 1s4z_vs_4s1z 1s4z_vs_5s1z 3s5z
# 1s4z_vs_6s1z 3s5z_vs_3s6z 3s5z_vs_3s7z 3s5z_vs_8s2z 3s5z_vs_4s6z 3s5z_vs_4s7z    3s5z_vs_4s8z

# MMM0: 1,2,7 vs 1,2,7
# MMM1: 1,1,7 vs 1,2,7  # + 1,7,2 vs 1,7,2
# MMM2: 1,2,7 vs 1,3,8
# MMM3: 1,2,8 vs 1,3,9
# MMM4: 1,3,8 vs 1,4,9   # + 1,8,3 vs 1,4,9
# MMM5: 1,3,8 vs 1,4,10
# MMM6: 1,3,8 vs 1,4,11
# MMM7: 1,3,8 vs 1,5,11
# MMM8: 1,2,7 vs 1,6,3  # 1,8,3 vs 1,5,12
# MMM9: 1,2,7 vs 1,6,4
# MMM10: 1,3,8 vs 1,5,12

def generate_curricula(envs, target_name, target_type_n, env_learned):
    envset = envs.copy()
    # env_learned = {}

    while True:
        max_potential = 0
        max_potential_name = ''
        for env, env_n in envset.items():
            sim_to_target = calculate_similarity(env_n, target_type_n)
            if env_learned:
                most_sim_name, sim_to_learned = most_similar(env_learned, env_n)
            else:
                sim_to_learned = 1
            scale = calculate_scale(env_n, target_type_n)
            potential = sim_to_target / sim_to_learned * scale
            if potential > max_potential:
                max_potential = potential
                max_potential_name = env
        env_learned[max_potential_name] = envset.pop(max_potential_name)
        print(f'max_potential_env: {max_potential_name}, potential:{max_potential}')
        if max_potential_name == target_name:
            break


def most_similar(envs, target_type_n):
    most_sim = 0
    most_sim_name = ''
    for name, type_n in envs.items():
        sim = calculate_similarity(type_n, target_type_n)
        # print(f'env_name:{name}, similarity:{sim}')
        if sim > most_sim:
            most_sim = sim
            most_sim_name = name
    return most_sim_name, most_sim


def calculate_similarity(env1, env2):
    # env1= [#type1, #type2 ...]
    # type1=enemy_s, type2=enemy_z, type3=ally_s, type4=ally_z
    similarity = 0
    for n1, n2 in zip(env1, env2):
        up = min(n1, n2)
        down = max(n1, n2)
        similarity += up / down
    return similarity


def calculate_scale(env1, env2):
    n_env1 = sum(env1) + 1
    n_env2 = sum(env2) + 1
    similarity = n_env2 / n_env1
    return similarity

def do_marines():
    # marines
    target_env_name = '7m_vs_9m'
    target_env_n = [7, 9]
    env_learned={'5m': [5, 5]}
    envs = {
        # '5m': [5, 5],
        '6m': [6, 6],
        '7m': [7, 7],
        '5m_vs_6m': [5, 6],
        '8m_vs_9m': [8, 9],
        '10m': [10, 10],
        '15m': [15, 15],
        '20m': [20, 20],
        '25m': [25, 25],
        '8m_vs_10m': [8, 10],
        '7m_vs_9m': [7, 9]
    }
    generate_curricula(envs, target_env_name, target_env_n, env_learned)

def do_sandz():
    # S and Z
    target_env_name='3s5z_vs_4s8z'
    target_env_n=[3,5,4,8]
    env_learned={'2s3z': [2,3,2,3]}
    envs = {
        '2s3z_vs_2s5z':[2,3,2,5],
        '3s5z_vs_8s2z':[3,5,8,2],
        # '3s6z_vs_4s8z':[3,6,4,8],
        '3s5z_vs_4s7z':[3,5,4,7],
        '1s4z_vs_6s1z':[1,4,6,1],
        '3s5z_vs_4s6z':[3,5,4,6],
        '3s5z_vs_3s7z':[3,5,3,7],
        '3s5z_vs_3s6z':[3,5,3,6],
        '2s3z_vs_2s4z':[2,3,2,4],
        '1s4z_vs_5s1z':[1,4,5,1],
        '3s5z':[3,5,3,5],
        '2s5z':[2,5,2,5],
        '3s4z':[3,4,3,4],
        '2s4z':[2,4,2,4],
        '1s4z_vs_4s1z':[1,4,4,1],
        '3s5z_vs_4s8z':[3,5,4,8]
    }
    generate_curricula(envs, target_env_name, target_env_n, env_learned)

def do_MMM():
    # MMM
    target_env_name = 'MMM10'
    target_env_n = [1, 3, 8, 1, 5, 12]
    env_learned={'MMM0': [1,2,7,1,2,7]}
    envs = {
        'MMM1': [1, 1, 7, 1, 2, 7],
        'MMM2': [1, 2, 7, 1, 3, 8],
        'MMM3': [1, 2, 8, 1, 3, 9],
        'MMM4': [1, 3, 8, 1, 4, 9],
        'MMM5': [1, 3, 8, 1, 4, 10],
        'MMM6': [1, 3, 8, 1, 4, 11],
        'MMM7': [1, 3, 8, 1, 5, 11],
        'MMM8': [1, 2, 7, 1, 6, 3],
        'MMM9': [1, 2, 7, 1, 6, 4],
        'MMM10': [1, 3, 8, 1, 5, 12],
    }
    generate_curricula(envs, target_env_name, target_env_n, env_learned)

if __name__ == '__main__':
    # do_marines()
    # do_sandz()
    do_MMM()
