from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib
from smac.env.starcraft2.maps import smac_maps

map_param_registry = {
    "2m": {
        "n_agents": 2,
        "n_enemies": 2,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "3m_vs_4m": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4m": {
        "n_agents": 4,
        "n_enemies": 4,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "4m_vs_5m": {
        "n_agents": 4,
        "n_enemies": 5,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "5m": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "6m": {
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 75,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "6m_vs_7m": {
        "n_agents": 6,
        "n_enemies": 7,
        "limit": 75,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "6m_vs_8m": {
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 75,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "7m": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 80,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "7m_vs_8m": {
        "n_agents": 7,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "7m_vs_9m": {
        "n_agents": 7,
        "n_enemies": 9,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "8m_vs_10m": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "9m": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "9m_vs_10m": {
        "n_agents": 9,
        "n_enemies": 10,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "9m_vs_11m": {
        "n_agents": 9,
        "n_enemies": 11,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "9m_vs_12m": {
        "n_agents": 9,
        "n_enemies": 12,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "10m": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "10m_vs_12m": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "11m": {
        "n_agents": 11,
        "n_enemies": 11,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "11m_vs_12m": {
        "n_agents": 11,
        "n_enemies": 12,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "11m_vs_13m": {
        "n_agents": 11,
        "n_enemies": 13,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "11m_vs_14m": {
        "n_agents": 11,
        "n_enemies": 14,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "11m_vs_15m": {
        "n_agents": 11,
        "n_enemies": 15,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "12m": {
        "n_agents": 12,
        "n_enemies": 12,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "12m_vs_13m": {
        "n_agents": 12,
        "n_enemies": 13,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "12m_vs_14m": {
        "n_agents": 12,
        "n_enemies": 14,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "12m_vs_15m": {
        "n_agents": 12,
        "n_enemies": 15,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "12m_vs_16m": {
        "n_agents": 12,
        "n_enemies": 16,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "13m": {
        "n_agents": 13,
        "n_enemies": 13,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "14m": {
        "n_agents": 14,
        "n_enemies": 14,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "15m": {
        "n_agents": 15,
        "n_enemies": 15,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "16m": {
        "n_agents": 16,
        "n_enemies": 16,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "17m": {
        "n_agents": 17,
        "n_enemies": 17,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "18m": {
        "n_agents": 18,
        "n_enemies": 18,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "19m": {
        "n_agents": 19,
        "n_enemies": 19,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "20m": {
        "n_agents": 20,
        "n_enemies": 20,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "21m": {
        "n_agents": 21,
        "n_enemies": 21,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "22m": {
        "n_agents": 22,
        "n_enemies": 22,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "23m": {
        "n_agents": 23,
        "n_enemies": 23,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "24m": {
        "n_agents": 24,
        "n_enemies": 24,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "26m": {
        "n_agents": 26,
        "n_enemies": 26,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "27m": {
        "n_agents": 27,
        "n_enemies": 27,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "28m": {
        "n_agents": 28,
        "n_enemies": 28,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "29m": {
        "n_agents": 29,
        "n_enemies": 29,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "30m": {
        "n_agents": 30,
        "n_enemies": 30,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    # add some s and z map
    "3s":{
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3s_vs_4s":{
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3s_vs_4s_rew3":{
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3z":{
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "3z_vs_4z":{
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "3z_vs_4z_rew3":{
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 100,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "3z_vs_3s":{
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "3z_vs_4s":{
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "3s_vs_4z_rew3": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3s_vs_5z_rew3": {
        "n_agents": 3,
        "n_enemies": 5,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    # sz maps
    "1s4z_vs_4s1z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "1s4z_vs_5s1z": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "1s4z_vs_6s1z": {
        "n_agents": 5,
        "n_enemies": 7,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s4z": {
        "n_agents": 6,
        "n_enemies": 6,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s5z": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s4z": {
        "n_agents": 7,
        "n_enemies": 7,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_2s4z": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_2s5z": {
        "n_agents": 5,
        "n_enemies": 7,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "2s3z_vs_2s6z": {
        "n_agents": 5,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_3s7z": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_3s8z": {
        "n_agents": 8,
        "n_enemies": 11,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_3s9z": {
        "n_agents": 8,
        "n_enemies": 12,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_4s6z": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_4s7z": {
        "n_agents": 8,
        "n_enemies": 11,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_4s8z": {
        "n_agents": 8,
        "n_enemies": 12,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_7s1z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_8s1z": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_8s2z": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s5z_vs_9s1z": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    # todo MM, MM2, MM3的unit_type_bits啥的还得调整
    "MM": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "MM",
    },
    "MM2": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 130,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "MM",
    },
    "MM3": {
        "n_agents": 8,
        "n_enemies": 10,
        "limit": 140,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "MM",
    },
    # MMM
    "MMM1": {
        "n_agents": 9,
        "n_enemies": 10,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM3": {
        "n_agents": 11,
        "n_enemies": 13,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM4": {
        "n_agents": 12,
        "n_enemies": 14,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM5": {
        "n_agents": 12,
        "n_enemies": 15,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM6": {
        "n_agents": 12,
        "n_enemies": 16,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM7": {
        "n_agents": 12,
        "n_enemies": 17,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM8": {
        "n_agents": 12,
        "n_enemies": 18,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM9": {
        "n_agents": 12,
        "n_enemies": 12,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM10": {
        "n_agents": 12,
        "n_enemies": 12,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM11": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM12": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    # some other maps
    "1o_10b_vs_1r": {
        "n_agents": 11,
        "n_enemies": 1,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_bane"
    },
    "1o_2r_vs_4r": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 50,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "overload_roach"
    },
    "bane_vs_hM": {
        "n_agents": 3,
        "n_enemies": 2,
        "limit": 30,
        "a_race": "Z",
        "b_race": "T",
        "unit_type_bits": 2,
        "map_type": "bZ_hM"
    },
}


smac_maps.map_param_registry.update(map_param_registry)

def get_map_params(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    return map_param_registry[map_name]


for name in map_param_registry.keys():
    globals()[name] = type(name, (smac_maps.SMACMap,), dict(filename=name))

# print(f"{__file__}")