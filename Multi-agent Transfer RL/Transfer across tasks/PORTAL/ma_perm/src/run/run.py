import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import sys
import numpy as np

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from envs.starcraft.StarCraft2Env import StarCraft2Env
from openpyxl import Workbook
import wandb

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return 4 + sc_env.shield_bits_ally + sc_env.unit_type_bits, sc_env.unit_type_bits


def get_map_unit_type_info(map_name):
    env = StarCraft2Env(map_name=map_name)
    obs_demo, state_demo = env.reset()
    print('obs_demo {}'.format(np.array(obs_demo).shape))
    print('state_demo {}'.format(np.array(state_demo).shape))
    # 2, {'enemy': [0, 0, 0, 1, 1], 'ally': [0, 0, 1, 1, 1]}
    return env.unit_type_bits, env.get_obs_component_unit_type()


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    # args.thread_num = 4
    if args.save_path:
        args.save_path = os.path.join(args.save_path, str(args.seed))
    th.set_num_threads(args.thread_num)
    # th.set_num_interop_threads(8)

    args.device = "cuda" if args.use_cuda else "cpu"

    # add wandb settings
    wandb_name = args.wandb_name
    tags = [args.env_args['map_name']]
    if args.checkpoint_path:
        tags.append("transfer")
    else:
        tags.append("train")
    tags.append(f"eps_start={args.epsilon_start}")
    tags.append(f"eps_finish={args.epsilon_finish}")
    tags.append(f"seed={args.seed}")
    tags.append(f"lr={args.lr}")
    if args.debug_dir:
        os.environ["WANDB_API_KEY"] = 'YOUR WANDB API'
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="YOUR PROJECT NAME", entity="YOUR ENTITY NAME", name=wandb_name, config=_config, tags=tags)
    else:
        wandb.init(project="YOUR PROJECT NAME", entity="YOUR ENTITY NAME", name=wandb_name, config=_config, tags=tags)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    testing_algorithms = ["vdn", "qmix", "dpn_vdn", "dpn_qmix", "api_vdn", "api_qmix",
                          "deepset_vdn", "deepset_qmix", "deepset_hyper_vdn", "deepset_hyper_qmix",
                          "updet_vdn", "updet_qmix", "vdn_DA", "qmix_DA",
                          "api_vdn_multihead", "api_vdn_multihead_noshare", "api_qmix_multihead", "api_vdn_residual", "api_qmix_residual",
                          "gnn_vdn", "gnn_qmix", 'api_vdn_multihead_dividehyper', 'api_qmix_multihead_dividehyper',
                          "api_vdn_multihead_relation", "api_vdn_multihead_relation_v0", "api_vdn_multihead_relation_v31",
                          "api_vdn_multihead_relation_v311", "api_vdn_multihead_relation_v32", "api_vdn_multihead_relation_v321",
                          "api_vdn_multihead_relation_v4", "api_vdn_multihead_relation_v33", "api_vdn_multihead_relation_v23",
                          'api_vdn_multihead_relation_v51','api_vdn_multihead_relation_v52','api_vdn_multihead_relation_v53',
                          "api_vdn_multihead_relation_v33_dividehyper", "dyan_attackunit_vdn"
                          ]
    env_name = args.env
    logdir = env_name
    if env_name in ["sc2"]:
        logdir = os.path.join("{}_{}-obs_aid={}-obs_act={}".format(
            logdir,
            args.env_args["map_name"],
            # int(args.env_args["reward_only_positive"]),
            int(args.obs_agent_id),
            int(args.obs_last_action),
        ))
    logdir = os.path.join(logdir,
                          "algo={}-agent={}".format(args.name, args.agent),
                          "env_n={}".format(
                              args.batch_size_run,
                          ))
    if args.name in testing_algorithms:
        if args.name in ["vdn_DA", "qmix_DA", ]:
            logdir = os.path.join(logdir,
                                  "{}-data_augment={}".format(
                                      args.mixer, args.augment_times
                                  ))
        elif args.name in ["gnn_vdn", "gnn_qmix"]:
            logdir = os.path.join(logdir,
                                  "{}-layer_num={}".format(
                                      args.mixer, args.gnn_layer_num
                                  ))
        elif args.name in ["vdn", "qmix", "deepset_vdn", "deepset_qmix", "dyan_attackunit_vdn"]:
            logdir = os.path.join(logdir,
                                  "mixer={}".format(
                                      args.mixer,
                                  ))
        elif args.name in ["deepset_hyper_vdn", "deepset_hyper_qmix"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-api_hyperdim={}".format(
                                      args.mixer,
                                      args.api_hyper_dim,
                                  ))
        elif args.name in ["api_vdn", "api_qmix"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-api_hyperdim={}-acti={}-out_acti={}".format(
                                      args.mixer,
                                      args.api_hyper_dim,
                                      args.api_hyper_activation,
                                      int(args.api_hyper_out_acti)
                                  ))
        elif args.name in ["api_vdn_multihead", "api_qmix_multihead", "api_vdn_multihead_noshare"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-api_hyperdim={}-acti={}-api_head={}".format(
                                      args.mixer,
                                      args.api_hyper_dim,
                                      args.api_hyper_activation,
                                      args.api_head_num,
                                  ))
        elif args.name in ["api_vdn_residual", "api_qmix_residual"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-api_hyperdim={}-residual_depth={}".format(
                                      args.mixer,
                                      args.api_hyper_dim,
                                      args.residual_depth,
                                  ))
        elif args.name in ["dpn_vdn", "dpn_qmix"]:
            logdir = os.path.join(logdir,
                                  "mac={}-mixer={}".format(
                                      args.mac,
                                      args.mixer),
                                  "rand_proj={}-auxi_loss={}-auxi_freq={}-auxi_coef={}-up_Q_stored={}".format(
                                      int(args.random_projection),
                                      int(args.auxiliary_loss),
                                      args.auxiliary_update_frequency,
                                      args.auxiliary_loss_coef,
                                      int(args.update_Q_using_stored_assignment),
                                  ),
                                  "emb_dim={}-k_ex={}-tau={}".format(
                                      args.assignment_net_dim,
                                      args.k_exchange,
                                      args.softmax_tau,
                                  ))
            if args.use_sinkhorn:
                logdir = os.path.join(logdir, "sink_iter={}-gum_noise={}".format(
                    args.sinkhorn_iters,
                    int(args.add_gumbel_noise),
                ))
            if args.mac == "permutation_mac":
                logdir = os.path.join(logdir,
                                      "reassign={}-{}".format(args.permute_condition, args.permute_condition_value))
        elif args.name in ['api_vdn_multihead_dividehyper', 'api_qmix_multihead_dividehyper', "api_vdn_multihead_relation_v33_divide_hyper"]:
            logdir = os.path.join(logdir,
                                  "mixer={}-api_hyperdim={}-acti={}-api_head={}".format(
                                      args.mixer,
                                      args.api_hyper_dim,
                                      args.api_hyper_activation,
                                      args.api_head_num,
                                  ))
        elif args.name in ['api_vdn_multihead_relation', 'api_vdn_multihead_relation_v31', 'api_vdn_multihead_relation_v311',
                           'api_vdn_multihead_relation_v32', 'api_vdn_multihead_relation_v321', "api_vdn_multihead_relation_v33",
                           "api_vdn_multihead_relation_v4", "api_vdn_multihead_relation_v0", "api_vdn_multihead_relation_v23",
                           'api_vdn_multihead_relation_v51', 'api_vdn_multihead_relation_v52', 'api_vdn_multihead_relation_v53',
                           ]:
            logdir = os.path.join(logdir,
                                  "mixer={}-api_hyperdim={}-acti={}-api_head={}".format(
                                      args.mixer,
                                      args.api_hyper_dim,
                                      args.api_hyper_activation,
                                      args.api_head_num,
                                  ))
    logdir = os.path.join(logdir,
                          "rnn_dim={}-2bs={}_{}-tdlambda={}-epdec_{}={}k".format(
                              args.rnn_hidden_dim,
                              args.buffer_size,
                              args.batch_size,
                              args.td_lambda,
                              args.epsilon_finish,
                              args.epsilon_anneal_time // 1000,
                          ))
    args.log_model_dir = logdir
    if args.run_type not in ['evaluate', 'collect', 'collect_raw', 'regret']:
        if args.use_tensorboard:
            if args.is_curriculum:
                tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), args.save_path, args.env_args["map_name"], "tb_logs")
            else:
                tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), args.local_results_path, "tb_logs")
            tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
            if args.name in testing_algorithms:  # add parameter config to the logger path！
                tb_exp_direc = os.path.join(tb_logs_direc, logdir, unique_token)
            logger.setup_tb(tb_exp_direc)

        # sacred is on by default
        logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger, wandb=wandb)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")

    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")
    os._exit(os.EX_OK)


def run_sequential(args, logger, wandb):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger, wandb=wandb)
    args.map_unit_type_bits, args.obs_component_divide = get_map_unit_type_info(args.env_args["map_name"])
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if args.env in ["sc2"]:
        args.n_enemies = env_info["n_enemies"]
        args.obs_ally_feats_size = env_info["obs_ally_feats_size"]
        args.obs_enemy_feats_size = env_info["obs_enemy_feats_size"]
        args.state_ally_feats_size = env_info["state_ally_feats_size"]
        args.state_enemy_feats_size = env_info["state_enemy_feats_size"]
        args.obs_component = env_info["obs_component"]
        args.state_component = env_info["state_component"]
        args.map_type = env_info["map_type"]
    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        # for reconstruction loss
        # "rec_obs":{"vshape": env_info["obs_shape"], "group": "agents"},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, wandb)

    if args.use_cuda:
        learner.cuda()

    if args.run_type:
        if args.run_type == 'train_rec':
            train_rec(args, logger, runner, buffer, learner)
            return

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

        if args.run_type == 'evaluate' or args.save_replay:
            assert args.save_path is not None
            evaluate_sequential(args, runner)
            return
        if args.run_type == 'collect':
            assert args.save_path is not None
            collect_sequential(args, runner)
            return
        if args.run_type == 'collect_raw':
            collect_raw_sample(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    if args.run_type == 'train_converge':
        runner.max_mean = 0
        runner.max_mean_step = 0
        runner.cnt = 0
        runner.MAX_CNT = 150
        runner.converge = False
    end_flag = False
    # for fine-tune
    freeze_flag = True
    while runner.t_env <= args.t_max and not end_flag:
        # Run for a whole episode at a time
        if freeze_flag and hasattr(args, 'step_for_finetune') and runner.t_env > args.step_for_finetune:
            learner.unfreeze()
            freeze_flag = False
        with th.no_grad():
            # t_start = time.time()
            episode_batch = runner.run(test_mode=False)
            if episode_batch.batch_size > 0:  # After clearing the batch data, the batch may be empty.
                buffer.insert_episode_batch(episode_batch)
            # print("Sample new batch cost {} seconds.".format(time.time() - t_start))

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)

        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            last_test_T = runner.t_env
            with th.no_grad():
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)
                    if hasattr(runner, 'converge') and runner.converge:
                        end_flag = True
                        # save converged model
                        model_save_time = runner.t_env
                        if args.save_path:
                            save_path = os.path.join(args.save_path, args.env_args["map_name"], 'model',
                                                     f'{args.env_args["map_name"]}_api_vdn_mh_aid=0_na={args.n_agents}',
                                                     str(runner.max_mean_step))
                            with open(f'{args.save_path}/{args.env_args["map_name"]}/model_path.txt', 'w') as f:
                                f.write(os.path.dirname(save_path))
                        else:
                            save_path = os.path.join(args.local_results_path, "models", args.log_model_dir, args.unique_token,
                                                     str(runner.max_mean_step))
                        # "results/models/{}".format(unique_token)
                        os.makedirs(save_path, exist_ok=True)
                        logger.console_logger.info("Saving models to {}".format(save_path))
                        learner.save_models(save_path)
                        break

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval or runner.t_env >= args.t_max):
            model_save_time = runner.t_env
            if args.save_path:
                save_path = os.path.join(args.save_path, args.env_args["map_name"], 'model',
                                         f'{args.env_args["map_name"]}_api_vdn_mh_aid=0_na={args.n_agents}',
                                         str(runner.t_env))
                # os.makedirs(save_path, exist_ok=True)
                # with open(f'{args.save_path}/{args.env_args["map_name"]}/model_path.txt', 'w') as f:
                #     f.write(os.path.dirname(save_path))
            else:
                save_path = os.path.join(args.local_results_path, "models", args.log_model_dir, args.unique_token,
                                     str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)


        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.log_stat("episode_in_buffer", buffer.episodes_in_buffer, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

    # flush
    sys.stdout.flush()
    time.sleep(10)

# train with reconstruction loss
def train_rec(args, logger, runner, buffer, learner):
    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        with th.no_grad():
            # t_start = time.time()
            episode_batch = runner.run(test_mode=False)
            if episode_batch.batch_size > 0:  # After clearing the batch data, the batch may be empty.
                buffer.insert_episode_batch(episode_batch)
            # print("Sample new batch cost {} seconds.".format(time.time() - t_start))

        if buffer.can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            last_test_T = runner.t_env
            with th.no_grad():
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval or runner.t_env >= args.t_max):
            model_save_time = runner.t_env
            if args.save_path:
                save_path = os.path.join(args.save_path, args.env_args["map_name"], 'model',
                                         f'{args.env_args["map_name"]}_api_vdn_mh_aid=0_mean_ne={args.n_enemies}',
                                         str(runner.t_env))
                with open(f'{args.save_path}/{args.env_args["map_name"]}/model_path.txt', 'w') as f:
                    f.write(os.path.dirname(save_path))
            else:
                save_path = os.path.join(args.local_results_path, "models", args.log_model_dir, args.unique_token,
                                         str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)


        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.log_stat("episode_in_buffer", buffer.episodes_in_buffer, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

    # flush
    sys.stdout.flush()
    time.sleep(10)

# evaluate the return
def evaluate_sequential(args, runner):
    won_rate_list = []
    return_list = []
    for _ in range(args.test_nepisode):
        _, episode_return, env_info = runner.run(test_mode=True)
        won_rate_list.append(env_info['battle_won'])
        return_list.append(episode_return)
    args.save_replay = True
    if args.save_replay:
        runner.save_replay()
    dict = {'won_list' : won_rate_list,
            'return_list' : return_list}
    policy = args.checkpoint_path.split('/')[-1]
    n = args.test_nepisode
    # curriculum
    policy_env_name= policy[:policy.find('_api')]
    save_file(f'{args.save_path}/{policy_env_name}/return/evaluate_env_{args.env_args["map_name"]}_{n}.npy',
              np.array(dict))


# collect evaluate data
def collect_sequential(args, runner):
    feature_buffers = []
    feature_rnn_buffers = []
    n = int(1e4)
    agent_mean=False
    # near_n = 5, ally = 4, enemy = 5
    near_n = -1
    c = 0
    return_list = []
    won_rate_list = []
    while c < n:
    # for _ in range(32):
        feature_buffer, feature_rnn_buffer, episode_return, env_info = runner.collect(test_mode=True, near_n=near_n, agent_mean=agent_mean)
        # feature_buffer.append(fb)
        feature_buffers.extend(feature_buffer)
        feature_rnn_buffers.extend(feature_rnn_buffer)
        # feature_buffers.append(feature_buffer)
        # feature_rnn_buffers.append(feature_rnn_buffer)
        won_rate_list.append(env_info['battle_won'])
        return_list.append(episode_return)
        c += len(feature_buffer)
        print(f'{c}/{n}')
    policy = args.checkpoint_path.split('/')[-1]
    # curriculum
    dict = {'won_list' : won_rate_list,
            'return_list' : return_list}
    policy_env_name= policy[:policy.find('_api')]
    if agent_mean:
        save_file(f'{args.save_path}/{policy_env_name}/mean_distance/{n}/{args.env_args["map_name"]}.npy',
                  np.array(feature_buffers[:n]))
        save_file(f'{args.save_path}/{policy_env_name}/mean_distance_rnn/{n}/{args.env_args["map_name"]}.npy',
                  np.array(feature_rnn_buffers[:n]))
    else:
        save_file(f'{args.save_path}/{policy_env_name}/distance/{n}/{args.env_args["map_name"]}.npy',
                  np.array(feature_buffers[:n]))
        save_file(f'{args.save_path}/{policy_env_name}/distance_rnn/{n}/{args.env_args["map_name"]}.npy',
                  np.array(feature_rnn_buffers[:n]))
    save_file(f'{args.save_path}/{policy_env_name}/return/{args.env_args["map_name"]}_{len(won_rate_list)}.npy',
                  np.array(dict))
    runner.close_env()

def collect_raw_sample(args, runner):
    n = int(2e4)
    c = 0
    batch_n = 0
    policy = args.checkpoint_path.split('/')[-1]
    policy_env_name= policy[:policy.find('_api')]
    while c < n:
        batch, total_filled = runner.collect_raw(test_mode=True)
        # feature_buffer.append(fb)
        c += total_filled * args.n_agents
        print(f'{c}/{n}')
        save_file(f'{args.save_path}/{policy_env_name}/distance/raw_datas/{n}/{args.env_args["map_name"]}/batch_{batch_n}.npy',
                  batch.data.transition_data)
        batch_n += 1
    runner.close_env()

def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config

def save_file(save_path, data):
    if not os.path.exists(save_path):
        recur_dir(save_path)
    np.save(save_path, data)

def recur_dir(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        recur_dir(dir_path)
        os.mkdir(dir_path)

