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

    mtTeacher = MTTeacher(cfg["teacher"], cfg["dir"]["base"], cfg["dir"]["sum"], cfg["seed"], load_buffer=False)
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

    for i in range(5):
        for p, env_test in zip(mtTeacher.policy_list, mtenv.env_test_list):
            eval_policy(p, env_test)
        eval_policy_mt(policy, mtTeacher, i, 5)
