import copy
import time

import torch as th
# from torch.multiprocessing import Pool
from multiprocessing import Pool
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from envs.matrix_game import print_matrix_status
from modules.mixers.api_qmix import APIQMixer
from modules.mixers.api_qmix_v0 import APIQMixer as APIQMixerV0
from modules.mixers.api_qmix_v0_hyper import APIQMixer as APIQMixerV0Hyper
from modules.mixers.api_qmix_v1_share_embed import APIQMixer as APIQMixerV1
from modules.mixers.api_qmix_v2_share_embed_2layer import APIQMixer as APIQMixerV2
from modules.mixers.api_qmix_v3 import APIQMixer as APIQMixerV3
from modules.mixers.api_qmix_v3_easy import APIQMixer as APIQMixerV3Easy
from modules.mixers.api_qmix_v3_easier import APIQMixer as APIQMixerV3Easier
from modules.mixers.api_qmix_v3_w1_origin import APIQMixer as APIQMixerV3W1Origin
from modules.mixers.api_qmix_v3_instancenorm import APIQMixer as APIQMixerV3_instancenorm
from modules.mixers.api_qmix_v3_layernorm import APIQMixer as APIQMixerV3_layernorm
from modules.mixers.api_qmix_v4 import APIQMixer as APIQMixerV4
from modules.mixers.api_qmix_v4_easy import APIQMixer as APIQMixerV4Easy
from modules.mixers.api_qmix_v5_hyper_w_share import APIQMixer as APIQMixerV5
from modules.mixers.api_qmix_v6_hyper_fully_share import APIQMixer as APIQMixerV6
from modules.mixers.api_qmix_v8_qhyperw import APIQMixer as APIQMixerV8
from modules.mixers.api_qmix_v9 import APIQMixer as APIQMixerV9
from modules.mixers.api_qmix_v10_layernorm import APIQMixer as APIQMixerV10_layernorm
from modules.mixers.api_qmix_v10_groupnorm import APIQMixer as APIQMixerV10_groupnorm
from modules.mixers.api_qmix_v10_instancenorm import APIQMixer as APIQMixerV10_instancenorm
from modules.mixers.api_qmix_v10_selfatten import APIQMixer as APIQMixerV10_selfatten
from modules.mixers.api_qmix_v11_nohyper_2w import APIQMixer as APIQMixerV11_nohyper_2w
from modules.mixers.api_qmix_v12_nohyper_2w2b import APIQMixer as APIQMixerV12_nohyper_2w2b
from modules.mixers.api_qmix_v13_nohyper_2b import APIQMixer as APIQMixerV13_nohyper_2b
from modules.mixers.api_qmix_v14_nohyper_1w2b import APIQMixer as APIQMixerV14_nohyper_1w2b
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num


def calculate_target_q(target_mac, batch, enable_parallel_computing=False, thread_num=4):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        # Set target mac to testing mode
        target_mac.set_evaluation_mode()
        target_mac_out = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
        return target_mac_out


def calculate_n_step_td_target(target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma, td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        target_max_qvals = target_mixer(target_max_qvals, batch["state"])

        if q_lambda:
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class NQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        elif args.mixer == "api_qmix":
            self.mixer = APIQMixer(args)
        elif args.mixer == "api_qmix_v0":
            self.mixer = APIQMixerV0(args)
        elif args.mixer == "api_qmix_v0_hyper":
            self.mixer = APIQMixerV0Hyper(args)
        elif args.mixer == "api_qmix_v1":  # 14.241K
            self.mixer = APIQMixerV1(args)
        elif args.mixer == "api_qmix_v2":  # 39.777K
            self.mixer = APIQMixerV2(args)
        elif args.mixer == "api_qmix_v3":  # 102.177K (5m_vs_6m)
            self.mixer = APIQMixerV3(args)
        elif args.mixer == "api_qmix_v3_easy":  # 58.273K (5m_vs_6m)  58.017K, 58.209K
            self.mixer = APIQMixerV3Easy(args)
        elif args.mixer == "api_qmix_v3_easier":  # 18.657K (5m_vs_6m)
            self.mixer = APIQMixerV3Easier(args)
        elif args.mixer == "api_qmix_v3_w1_origin":  # 58.273K (5m_vs_6m)  58.017K
            self.mixer = APIQMixerV3W1Origin(args)
        elif args.mixer == "api_qmix_v3_instancenorm":  # 58.273K (5m_vs_6m)  58.017K
            self.mixer = APIQMixerV3_instancenorm(args)
        elif args.mixer == "api_qmix_v3_layernorm":  # 58.273K (5m_vs_6m)  58.017K
            self.mixer = APIQMixerV3_layernorm(args)
        elif args.mixer == "api_qmix_v4":  # 57.345K
            self.mixer = APIQMixerV4(args)
        elif args.mixer == "api_qmix_v4_easy":  # 57.345K
            self.mixer = APIQMixerV4Easy(args)
        elif args.mixer == "api_qmix_v5":  # 72.481K
            self.mixer = APIQMixerV5(args)
        elif args.mixer == "api_qmix_v6":  # 100.449K
            self.mixer = APIQMixerV6(args)
        elif args.mixer == "api_qmix_v8":  # 33.665K
            self.mixer = APIQMixerV8(args)
        elif args.mixer == "api_qmix_v9":  # 33.761K
            self.mixer = APIQMixerV9(args)
        elif args.mixer == "api_qmix_v10_layernorm":  # 33.761K
            self.mixer = APIQMixerV10_layernorm(args)
        elif args.mixer == "api_qmix_v10_groupnorm":  # 33.761K
            self.mixer = APIQMixerV10_groupnorm(args)
        elif args.mixer == "api_qmix_v10_instancenorm":  # 33.761K
            self.mixer = APIQMixerV10_instancenorm(args)
        elif args.mixer == "api_qmix_v10_selfatten":  # 34.337K
            self.mixer = APIQMixerV10_selfatten(args)
        elif args.mixer == "api_qmix_v11_nohyper_2w":  # 15.585K
            self.mixer = APIQMixerV11_nohyper_2w(args)
        elif args.mixer == "api_qmix_v12_nohyper_2w2b":  # 0.225K
            self.mixer = APIQMixerV12_nohyper_2w2b(args)
        elif args.mixer == "api_qmix_v13_nohyper_2b":  # 42.752K
            self.mixer = APIQMixerV13_nohyper_2b(args)
        elif args.mixer == "api_qmix_v14_nohyper_1w2b":  # 42.752K
            self.mixer = APIQMixerV14_nohyper_1w2b(args)
        else:
            raise "mixer error"

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0

        self.enable_parallel_computing = (not self.args.use_cuda) and getattr(self.args, 'enable_parallel_computing',
                                                                              True)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            # Multiprocessing pool for parallel computing.
            # ctx = th.multiprocessing.get_context("spawn")
            # self.pool = ctx.Pool()
            self.pool = Pool(1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.enable_parallel_computing:
            target_mac_out = self.pool.apply_async(
                calculate_target_q,
                (self.target_mac, batch, True, self.args.thread_num)
            )

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            if self.enable_parallel_computing:
                target_mac_out = target_mac_out.get()
            else:
                target_mac_out = calculate_target_q(self.target_mac, batch)

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            assert getattr(self.args, 'q_lambda', False) == False
            if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                targets = self.pool.apply_async(
                    calculate_n_step_td_target,
                    (self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                     self.args.td_lambda, True, self.args.thread_num, False, None)
                )
            else:
                targets = calculate_n_step_td_target(
                    self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                    self.args.td_lambda
                )

        # Set mixing net to training mode
        self.mixer.train()
        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
            targets = targets.get()

        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)
        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask
        mask_elems = mask.sum()
        loss = masked_td_error.sum() / mask_elems

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # For log
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def __del__(self):
        if self.enable_parallel_computing:
            self.pool.close()
