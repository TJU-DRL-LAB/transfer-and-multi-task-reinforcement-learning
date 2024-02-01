# coding=UTF-8
import random

import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
from run import REGISTRY as run_REGISTRY

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = join(dirname(dirname(abspath(__file__))))


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    seed = _config['seed']
    # seed = 75981140  # for debug only
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    # th.cuda.manual_seed_all(config["seed"])
    th.backends.cudnn.deterministic = True  # cudnn


    config['env_args']['seed'] = seed
    # run
    if "use_per" in _config and _config["use_per"]:
        run_REGISTRY['per_run'](_run, config, _log)
    else:
        run_REGISTRY[_config['run']](_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=') + 1:].strip()
            break
    return result

def find_params(params, key):
    value = None
    for param in params:
        if key in param:
            value = ''.join(param.split('=')[1:])
    assert value is not None
    return value

if __name__ == '__main__':
    # args = parse_args()
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    run_type = find_params(params, 'run_type=')
    debug_dir = find_params(params, 'debug_dir=')
    checkpoint_path = find_params(params, 'checkpoint_path=')
    save_path = find_params(params, 'save_path=')
    seed = find_params(params, 'seed=')
    wandb_name = find_params(params, 'wandb_name=')
    lr = find_params(params, 'lr=')
    save_path = os.path.join(save_path, seed)
    aggregation = find_params(params, 'aggregation=')
    config_dict['aggregation'] = aggregation
    assert aggregation in ['mean', 'sum']
    is_curriculum = find_params(params, 'is_curriculum=')
    config_dict['is_curriculum'] = is_curriculum
    if checkpoint_path:
        algo_config_name = find_params(params, '--config=')
        assert algo_config_name.endswith('reload')
    if seed:
        config_dict['seed'] = seed
    if debug_dir == "True":
        config_dict["local_results_path"] = "debug_dir"
    config_dict['cof']=find_params(params, 'cof=')
    config_dict["debug_dir"] = "True"
    assert run_type in ['evaluate', 'collect', 'collect_raw', 'train', 'regret', 'train_rec', 'train_converge']
    config_dict['run_type'] = run_type
    config_dict['save_path'] = save_path
    config_dict['wandb_name'] = wandb_name
    config_dict['lr'] = lr

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)
    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name'])
    local_results_path = parse_command(params, "local_results_path", config_dict['local_results_path'])
    if is_curriculum == "True":
        file_obs_path = join(results_path, save_path, map_name, "sacred")
    else:
        file_obs_path = join(results_path, local_results_path, "sacred", map_name, algo_name)

    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

    # flush
    sys.stdout.flush()
