import copy
import time
from multiprocessing import Pool

import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer as QMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num


def calculate_target(target_mac, target_mixer, batch, cur_max_actions,
                     using_stored_assignment, mixer, use_q_lambda, gamma, td_lambda):
    """
    Using multiprocessing to facilitate training.
    :param target_mac:
    :param batch:
    :return:
    """
    th.set_num_threads(3)
    # th.set_num_interop_threads(8)
    with th.no_grad():
        # Calculate the Q-Values necessary for the target
        # TODO: %%%%%%%%%%%%%%%% 并行：开另一个进程计算 target %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        target_mac_out = []
        target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = target_mac.forward(
                batch, t=t,
                sample_action=False,
                test_mode=False,
                t_env=-1
            )
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time [batch, traj, n_agents, n_actions]
        # TODO: %%%%%%%%%%%%%%%% 并行：开另一个进程计算 target %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # Max over target Q-Values:
        # For the reason of using Double q learning, the Q_tot_target calculation cannot be moved to the multiprocessing part.
        target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)  # [batch, traj, n_agents]

        if mixer == "qmix":
            target_q_states = target_mac.transform_states(batch["state"])
        else:
            target_q_states = batch["state"]

        # Calculate n-step Q-Learning targets
        target_max_qvals = target_mixer(target_max_qvals, target_q_states)

        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        if use_q_lambda:
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)  # [batch, traj, n_agents]
            qvals = target_mixer(qvals, target_q_states)

            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                             gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                              gamma, td_lambda)
        return targets


def calculate_permuted_q_for_target(target_mac, batch, using_stored_assignment, mixer):
    th.set_num_threads(3)
    # th.set_num_interop_threads(8)
    with th.no_grad():
        if mixer == "qmix":
            target_q_states = target_mac.transform_states(batch["state"])
        else:
            target_q_states = batch["state"]

        return target_mac.prepare_permuted_q_for_training(batch, is_target=True), target_q_states


class PermutationQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = QMixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimizer = Adam(params=self.params, lr=args.lr)
        else:
            self.optimizer = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # Only used for clear gradients for the individual-Q net.
        self.q_net_params = list(self.mac.q_net_parameters())
        # self.q_net_optimizer = Adam(params=self.q_net_params, lr=self.args.lr)
        self.assign_net_params = list(self.mac.transform_parameters())
        self.assign_net_optimizer = Adam(params=self.assign_net_params, lr=self.args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        # Multiprocessing pool for parallel computing.
        self.pool = Pool()
        self.train_t = 0
        self.avg_time = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        # terminated字段：Game真正结束了，而非因为超过了episode_limit而结束（SMAC默认，不管哪个结束都是结束）
        terminated = batch["terminated"][:, :-1].float()  # 前 200 步，是否真正结束
        mask = batch["filled"][:, :-1].float()  # 前 200 步，填充了的
        # TODO: terminated是执行action后是否结束(即看的是t+1)，但是存储的时候存在了时刻t，因此要向前错1位
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # 把已经结束了的位置->置零，不再更新
        avail_actions = batch["avail_actions"]

        # %%%%%%%%%%%%%% Auxiliary-loss %%%%%%%%%%%%%
        # episode_per_process = episode_num // self.args.batch_size_run
        # train_auxiliary_flag = self.args.auxiliary_loss and not self.args.random_projection and episode_per_process % self.args.auxiliary_update_frequency == 0
        # if train_auxiliary_flag:

        # Facilitate training.
        # Freeze the parameters of individual-Q. We only update the assignment parameters here.
        self._freeze_q()

        # _mac_out = self.mac.prepare_permuted_q_for_training(batch, )[:, :-1]
        _mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(
                batch, t=t, sample_action=False,
                test_mode=False,  # use gumbel softmax sampling
                t_env=-1
            )
            _mac_out.append(agent_outs)
        _mac_out = th.stack(_mac_out, dim=1)  # Concat over time

        _mac_out[avail_actions == 0] = -9999999  # mask out invalid actions
        # TODO: 因为这里的优化是最大化sum(Q)，也就是最大化individual Q，所以这次优化后，不会改变 cur_max_actions
        greedy_q_values, _cur_max_actions = _mac_out.max(dim=-1, keepdim=False)  # [bs, T, n_agents]
        greedy_q_tot = greedy_q_values[:, :-1].sum(dim=-1, keepdim=True)  # [bs, T-1, 1]
        mask = mask.expand_as(greedy_q_tot)
        assignment_loss = - (greedy_q_tot * mask).sum() / mask.sum()

        # (4) zero gradients
        self.assign_net_optimizer.zero_grad()
        # (5) Backward Auxiliary-loss
        assignment_loss.backward()
        transform_grad_norm = th.nn.utils.clip_grad_norm_(self.assign_net_params, self.args.grad_norm_clip)
        # (6) Clear the gradients of 'individual-Q Net' (only preserve the gradients of 'assignment Net').
        self.assign_net_optimizer.step()
        # print("Episode-{}: assign_loss={}".format(episode_num, assignment_loss.item()))

        # TODO: %%%%%%%%%%%%%%%% 并行：开另一个进程计算 target %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        target_mac_out = self.pool.apply_async(
            calculate_target,
            (
                self.target_mac, self.target_mixer, batch, _cur_max_actions.unsqueeze(-1),
                self.args.update_Q_using_stored_assignment, self.args.mixer, getattr(self.args, 'q_lambda', False),
                self.args.gamma, self.args.td_lambda
            )
        )
        # TODO: %%%%%%%%%%%%%%%% 并行：开另一个进程计算 target %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # mac_out = self.mac.prepare_permuted_q_for_training(batch)

        # Calculate estimated Q-Values
        self._unfreeze_q()
        self.mac._set_train_mode()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(
                batch, t=t,
                sample_action=False,
                test_mode=False,
                t_env=-1
            )
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(
            3)  # Remove the last dim: [batch, traj-1, n_agents]

        # Mixer
        if self.args.mixer == "qmix":
            q_states = self.mac.transform_states(batch["state"][:, :-1])
        else:
            q_states = batch["state"][:, :-1]
        chosen_action_qvals = self.mixer(chosen_action_qvals, q_states)
        # TODO: %%%%%%%%%%%%%%%% 并行：开另一个进程计算 target %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        targets = target_mac_out.get()
        # TODO: %%%%%%%%%%%%%%%% 并行：开另一个进程计算 target %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # %%%%%%%%%%%%%% TD-loss %%%%%%%%%%%%%
        td_error = (chosen_action_qvals - targets.detach())
        td_error = 0.5 * td_error.pow(2)
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        td_loss = masked_td_error.sum() / mask.sum()

        # %%%%%%%%%%%%%% Optimise %%%%%%%%%%%%%
        # (1) Clear all gradients of 'assignment Net' and 'individual-Q Net'.
        self.optimizer.zero_grad()
        # (2) Backward TD-loss.
        td_loss.backward(retain_graph=False)
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        # (3) Apply the accumulated gradient.
        self.optimizer.step()
        # print("Episode-{}: td_loss={}".format(episode_num, td_loss.item()))

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))

        # TODO: target update interval 会不会影响？？？？
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # if train_auxiliary_flag:
            self.logger.log_stat("assignment_loss", assignment_loss.item(), t_env)
            self.logger.log_stat("assignment_grad_norm", transform_grad_norm, t_env)
            self.logger.log_stat("loss_td", td_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _unfreeze_q(self):
        for pram in self.q_net_params:
            pram.requires_grad = True

    def _freeze_q(self):
        for pram in self.q_net_params:
            pram.requires_grad = False
            pram.grad = None

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
        th.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        print("Load model from {}".format(path))
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimizer.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def __del__(self):
        self.pool.close()
