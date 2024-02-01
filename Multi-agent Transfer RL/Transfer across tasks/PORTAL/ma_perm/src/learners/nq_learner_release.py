import copy
import time
import numpy as np
import torch
import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
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
            if type(target_agent_outs) == tuple:
                target_agent_outs = target_agent_outs[0]
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
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()


class NQLearner:
    def __init__(self, mac, scheme, logger, args, wandb):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.wandb = wandb
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        # elif args.mixer == "api_qmix":
        #     self.mixer = APIQMixer(args)
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
                                                                              False)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            from multiprocessing import Pool
            # Multiprocessing pool for parallel computing.
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
        # todo mask_rec 是否需要删去最后的时间维？
        mask_rec = batch["filled"].float()
        mask_rec[:, 1:] = mask_rec[:, 1:] * (1 - terminated)
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
        obs_rec_list = []
        obs_label_list = []
        obs_emb_list = []
        rec_flag = False
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            if type(agent_outs) == tuple:
                rec_flag = True
                obs_emb = agent_outs[2]
                rec_tuple = agent_outs[1]
                agent_outs = agent_outs[0]
                obs_emb_list.append((obs_emb))
                obs_rec_list.append(rec_tuple[0])
                obs_label_list.append(rec_tuple[1])
            mac_out.append(agent_outs)
        if rec_flag:
            # 维度 len(list) = t, list[0].shape = (batch, n_agents * 64)
            obs_emb_list = obs_emb_list[:-1]
            obs_emb_list = th.stack(obs_emb_list, dim=1)
            obs_rec_list = th.stack(obs_rec_list, dim=1)
            obs_label_list = th.stack(obs_label_list, dim=1)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # TODO: double DQN action, COMMENT: do not need copy
        mac_out[avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            if self.enable_parallel_computing:
                target_mac_out = target_mac_out.get()
            else:
                target_mac_out = calculate_target_q(self.target_mac, batch)

            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            mac_out_detach = mac_out
            # mac_out_detach[avail_actions == 0] = -9999999
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

        mask_rec_r = mask
        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        loss = masked_td_error.sum() / mask_elems

        # reconstruction loss
        if rec_flag:
            # action.shape=(bs, t-1, n_agent, 1), obs_emb_list.shape=(bs, t-1, n_agent * 64)
            bs = obs_emb_list.shape[0]
            t = obs_emb_list.shape[1]
            bst = bs * t
            # view无法在原数组上操作，因为不满足连续性条件
            action_t = actions.clone()
            action_t = action_t.view(bst, -1)
            rec_r = self.mac.get_rec_r(obs_emb_list.view(bst, -1), action_t)

            mask_rec = mask_rec.expand_as(obs_label_list)
            rec_loss_fn = th.nn.MSELoss(size_average=False)
            mask_rec_elems = mask_rec.sum()
            # 乘以mask
            obs_rec_list_mask = obs_rec_list * mask_rec
            obs_label_list_mask = obs_label_list * mask_rec
            rec_loss = rec_loss_fn(obs_rec_list_mask, obs_label_list_mask)
            rec_loss = rec_loss / mask_rec_elems

            # rewards.shape=(bs, t, 1)

            rec_r = rec_r.view(bs, t, -1)
            rec_r_mask = rec_r * mask_rec_r
            rewards_mask = mask_rec_r * rewards
            rec_r_loss = rec_loss_fn(rec_r_mask, rewards_mask)
            mask_rec_r_elems = mask_rec_r.sum()
            rec_r_loss = rec_r_loss / mask_rec_r_elems
            loss = loss + rec_loss + rec_r_loss

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
            self.wandb.log({"loss_td": loss.item(), "t_env": t_env})
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.wandb.log({"grad_norm":grad_norm, "t_env":t_env})
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.wandb.log({"td_error_abs":td_error_abs, "t_env":t_env})
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.wandb.log({"q_taken_mean":q_taken_mean, "t_env":t_env})
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.wandb.log({"target_mean":target_mean, "t_env":t_env})
            if rec_flag:
                self.logger.log_stat("reconstruction loss", rec_loss, t_env)
                self.wandb.log({"reconstruction loss":rec_loss, "t_env":t_env})
                self.logger.log_stat("reconstruction r loss", rec_r_loss, t_env)
                self.wandb.log({"reconstruction r loss":rec_r_loss, "t_env":t_env})
            self.log_stats_t = t_env

    def get_regret(self, batch: EpisodeBatch):
        # todo 改为mask用法
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        with th.no_grad():
            self.mac.set_train_mode()
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time
            # TODO: double DQN action, COMMENT: do not need copy
            mac_out[avail_actions == 0] = -9999999
            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
            target_mac_out = calculate_target_q(self.target_mac, batch)

            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            mac_out_detach = mac_out
            # mac_out_detach[avail_actions == 0] = -9999999
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
            # td_error, shape like [batch, episode_length, 1]
            td_error = (chosen_action_qvals - targets)
            td_error_abs = torch.abs(td_error)
            td_error_clip = torch.clamp(td_error, min=0)

            mask = mask.expand_as(td_error)
            masked_td_error = td_error * mask
            masked_td_error_abs = td_error_abs * mask
            masked_td_error_clip = td_error_clip * mask

            mask_elems = mask.sum()
            loss = masked_td_error.sum() / mask_elems
            loss_abs = masked_td_error_abs.sum() / mask_elems
            loss_clip = masked_td_error_clip.sum() / mask_elems
        return loss, loss_abs, loss_clip


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
        # 如果用VDN，直接注释下面
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))

        if self.args.name == 'api_vdn_multihead_noshare':
            # 如果是fine-tune的话，直接重新初始化新的优化器
            if self.args.optimizer == 'adam':
                base_params = filter(lambda p: not p.requires_grad, self.mac.agent.parameters())
                output_params = filter(lambda p: p.requires_grad, self.mac.agent.parameters())
                self.optimiser = Adam([
                    {'params': base_params, 'lr': self.args.lr * 0},
                    {'params': output_params, 'lr': self.args.lr},
                ])
            else:
                raise Exception('Currently, only adam optimiser is supported!')
        else:
            # 普通情况直接reload保存的optimiser
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.param_groups[0]['lr'] = self.args.lr

    def unfreeze(self):
        assert self.args.name == 'api_vdn_multihead_noshare'
        # fine-tune最后的output layer一定步数后，unfreeze前面的表征层
        for p in self.mac.agent.parameters():
            p.requires_grad = True
        # 前面的表征层，学习率不再是0
        self.optimiser.param_groups[0]['lr'] = self.args.lr


    def __del__(self):
        if self.enable_parallel_computing:
            self.pool.close()
