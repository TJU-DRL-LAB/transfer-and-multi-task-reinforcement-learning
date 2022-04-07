import time

from gym_extensions.continuous import mujoco
import argparse
import numpy as np
import torch
from agent.td3mt import TD3MT
from utils.mtenv import MTTeacher
from utils.logger import EnvLogger
from utils.preprocess import load_cfg


def eval_policy_mt(policy, mtTeacher, step, eval_episodes=3):
    mtenv = mtTeacher.mtenv
    for idx, env_test in enumerate(mtenv.env_test_list):
        avg_reward = 0.
        for i in range(eval_episodes):
            state, done = env_test.reset(), False
            while not done:
                state_pad = mtenv.pad_state(state)
                action_pad = policy.select_action(state_pad)
                action = mtenv.clip_action(idx, action_pad)
                state, r, done, _ = env_test.step(action * mtenv.max_action_list[idx])
                avg_reward += r

        avg_reward /= eval_episodes

        print("---------------------------------------")
        stat = str("Evaluation {} over {} reward: {:.3f} in {}"
                   .format(mtenv.env_name_list[idx], eval_episodes, avg_reward, step))
        logger.info(stat, extra={'label': 'eval'})
        mtTeacher.summer_list[idx].add_scalar("EpReward-Eval", avg_reward, step)
        mtTeacher.summer_list[idx].flush()
        print("---------------------------------------")


def eval_policy(policy, env_eval, eval_episodes=10):
    avg_reward = 0.
    for i in range(eval_episodes):
        state, done = env_eval.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, r, done, _ = env_eval.step(action * env_eval.action_space.high[0])
            avg_reward += r

    avg_reward /= eval_episodes

    print("***************************************")
    stat = str("Teacher {} over {} reward: {:.3f} in 0"
               .format(env_eval.unwrapped.spec.id, eval_episodes, avg_reward))
    logger.info(stat, extra={'label': 'eval'})
    print("***************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run flags")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cfg')
    parser.add_argument('--dir')
    parser.add_argument('--name')
    args = parser.parse_args()

    cfg = load_cfg(args.cfg, args.dir, task=args.name, seed=args.seed)

    train_flag = cfg["flag"] == "train"

    logger = EnvLogger(name="agent")
    logger.add_stream_handler()
    logger.add_file_handler(cfg["dir"]["log"], 'event.log')
    logger = logger.logger

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # todo fit for each env
    max_action = 1.0

    mtTeacher = MTTeacher(cfg["teacher"], cfg["dir"]["base"], cfg["dir"]["sum"], cfg["seed"], load_buffer=True)
    mtenv = mtTeacher.mtenv
    num_env = len(cfg["teacher"])

    kwargs = {
        "discount": cfg["train"]["gamma"],
        "tau": cfg["train"]["tau"],
        "policy_noise": cfg["noise"]["expl_noise"],
        "noise_clip": cfg["noise"]["noise_clip"],
        "policy_freq": cfg["noise"]["policy_freq"],
        "max_action": max_action,
        "state_dim": mtTeacher.mtenv.state_dim,
        "action_dim": mtTeacher.mtenv.action_dim,
        "num_env": num_env
    }

    policy = TD3MT(**kwargs)

    if "load" in cfg and cfg["load"]:
        policy.load(cfg["dir"]["base"] + "/" + cfg["load"])
        print("Restore Model from", cfg["load"])

    for p, env_test in zip(mtTeacher.policy_list, mtenv.env_test_list):
        eval_policy(p, env_test)
    eval_policy_mt(policy, mtTeacher, 0)

    btime = time.time()

    for st in range(int(cfg["step"]["training"]["offline"])):
        for idx in range(num_env):
            teacher, env, replay, summer = mtTeacher.get_bundle(idx)
            loss = policy.train_mt(idx, teacher, replay, cfg["train"]["batch_size"], is_offline=True)
            if (st + 1) % cfg["step"]["evaluation"] == 0:
                if loss:
                    if loss[0]:
                        summer.add_scalar("loss_actor", loss[0], st + 1)
                    if loss[1]:
                        summer.add_scalar("loss_critic", loss[1], st + 1)

        if (st + 1) % cfg["step"]["saving"] == 0:
            policy.save("{}/{}-{}".format(cfg["dir"]["mod"], 'mt', st + 1))

        if (st + 1) % cfg["step"]["evaluation"] == 0:
            eval_policy_mt(policy, mtTeacher, st + 1)
            print('Time Cost: ', time.time() - btime)
            btime = time.time()

    policy.sync_target_network()
    print("Finishing Offline Training, Start Online Training")
    mtTeacher.reset_all_replay()

    states = [mtenv.get_env(idx).reset() for idx in range(num_env)]
    dones = [False for _ in range(num_env)]
    ep_steps = [0 for _ in range(num_env)]
    ep_rewards = [0 for _ in range(num_env)]
    btime = time.time()

    for st in range(int(cfg["step"]["training"]["offline"]),
                    int(cfg["step"]["training"]["online"] + int(cfg["step"]["training"]["offline"]))):

        for idx in range(num_env):
            ep_steps[idx] += 1
            teacher, env, replay, summer = mtTeacher.get_bundle(idx)

            state = states[idx]
            state_pad = mtenv.pad_state(state)

            noise = np.random.normal(0, max_action * cfg["noise"]["expl_noise"], size=mtenv.action_dim)
            action_pad = policy.select_action(state_pad)
            action_pad = (action_pad + noise).clip(-max_action, max_action)
            action = mtenv.clip_action(idx, action_pad)

            next_state, reward, dones[idx], _ = env.step(action * mtenv.max_action_list[idx])
            done_bool = float(dones[idx]) if ep_steps[idx] < env._max_episode_steps else 0

            replay.add(state, action, next_state, reward, done_bool)

            states[idx] = next_state
            ep_rewards[idx] += reward

            loss = None
            if st >= cfg["step"]["observation"] + int(cfg["step"]["training"]["offline"]):
                loss = policy.train_mt(idx, teacher, replay, cfg["train"]["batch_size"], is_offline=False)

            if dones[idx]:
                stat = "Name {} " \
                       "Total T: {} " \
                       "Episode T: {} " \
                       "Reward: {:.3f} " \
                    .format(mtenv.env_name_list[idx], st + 1, ep_steps[idx], ep_rewards[idx])
                logger.info(stat, extra={'label': 'train'})
                summer.add_scalar("EpReward-Train", ep_rewards[idx], st + 1)
                if loss:
                    if loss[0]:
                        summer.add_scalar("loss_actor", loss[0], st + 1)
                    if loss[1]:
                        summer.add_scalar("loss_critic", loss[1], st + 1)

                states[idx], dones[idx] = env.reset(), False
                ep_steps[idx] = 0
                ep_rewards[idx] = 0

        if (st + 1) % cfg["step"]["saving"] == 0:
            policy.save("{}/{}-{}".format(cfg["dir"]["mod"], 'mt', st + 1))

        if (st + 1) % cfg["step"]["evaluation"] == 0:
            eval_policy_mt(policy, mtTeacher, st + 1)
            print('Time Cost: ', time.time() - btime)
            btime = time.time()
