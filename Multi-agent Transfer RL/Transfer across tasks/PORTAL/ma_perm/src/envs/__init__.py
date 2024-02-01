from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .starcraft import StarCraft2EnvVisible
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv
except:
    gfootball = False


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2_visible"] = partial(env_fn, env=StarCraft2EnvVisible)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if sys.platform == "linux":
    # os.environ.setdefault("SC2PATH",
    #                       os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
    os.environ.setdefault("SC2PATH", "~/rl_dir/pymarl/3rdparty/StarCraftII")
