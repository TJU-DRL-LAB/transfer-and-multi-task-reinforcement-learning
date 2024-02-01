import itertools

import heapq
import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.epsilon_schedules import DecayThenFlatSchedule
from modules.agents import REGISTRY as agent_REGISTRY
from .basic_controller import BasicMAC
from utils.th_utils import get_parameters_num


def onehot_from_logits(logits, dim):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    index = logits.max(dim, keepdim=True)[1]
    y_hard = th.zeros_like(logits, memory_format=th.legacy_contiguous_format).scatter_(dim, index, 1.0)
    return y_hard


# Limit values suitable for use as close to a -inf logit. These are useful
# since -inf / inf cause NaNs during backprop.
FLOAT_MIN = -3.4e38
FLOAT_MAX = 3.4e38


class TransformationAgent(nn.Module):
    def __init__(self, args):
        super(TransformationAgent, self).__init__()
        self.args = args
        n_allies, ally_feats_dim = args.obs_ally_feats_size
        n_enemies, enemy_feats_dim = args.obs_enemy_feats_size
        self.enemy_assignment_policy = nn.Sequential(
            nn.Linear(enemy_feats_dim, args.assignment_net_dim),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(args.assignment_net_dim, n_enemies),
        )
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")

    def _generate_action_prob(self, logit, batch_invalid_actions, deterministic_mode):
        # mask_inf = th.clamp(th.log(1 - batch_invalid_actions.float()), FLOAT_MIN,
        #                     FLOAT_MAX)  # 2-agent example: [[-inf, 0], [0, -inf]]
        negative_mask = batch_invalid_actions * FLOAT_MIN
        logit_masked = logit + negative_mask
        if deterministic_mode:
            return onehot_from_logits(logit_masked, dim=-1).detach()
        else:
            return F.gumbel_softmax(logit_masked, tau=self.args.softmax_tau, hard=True, dim=-1)

    def forward(self, ally_feats, enemy_feats, is_sample_action, is_target_net, t_env):
        """
        :param ally_feats:  [bs, n_agents, n_allies, ally_fea_dim]
        :param enemy_feats:  [bs, n_agents, n_enemies, enemy_fea_dim]
        :return:  assignment: [bs, n_agents, n_enemies, n_position]
        """
        # Online sampling or target_net for offline training
        # deterministic_mode = is_sample_action or (not is_sample_action and is_target_net)
        deterministic_mode = False  # TODO:

        # bs, n_agents, n_allies, ally_fea_dim = ally_feats.size()
        bs, n_agents, n_enemies, enemy_fea_dim = enemy_feats.size()

        enemy_logits = self.enemy_assignment_policy(enemy_feats)  # [bs, n_agents, n_enemies, n_enemy_position]

        # the position order (learned order) is stable
        # %%%%%%%%%%%%%%%%%%%%%%%%% Enemies %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        _invalid_enemy_mask = th.zeros(size=[bs, n_agents, n_enemies], dtype=th.float,
                                       device=enemy_feats.device)  # [bs, n_agents, n_enemies]  column order
        permutation_matrices = []
        # For each column, compute the assignment probability
        for position_idx in range(n_enemies):  # over n_position, the position order (learned order) is stable
            logit = enemy_logits[:, :, :, position_idx]  # [bs, n_agents, n_enemies]  column order
            _enemy_prob = self._generate_action_prob(logit, _invalid_enemy_mask,
                                                     deterministic_mode)  # [bs, n_agents, n_enemies]
            _invalid_enemy_mask = _invalid_enemy_mask + _enemy_prob.detach()
            permutation_matrices.append(_enemy_prob)
        permutation_matrices = th.stack(permutation_matrices, dim=2)  # [bs, n_agents, n_position, n_enemies]
        # %%%%%%%%%%%%%%%%%%%%%%%%% Enemies %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # TODO Exploration: 对 permutation matrix进行置换： online在线采样（不管是train还是test）都应该是deterministic的permutation，而离线训练时，适当增加探索
        # TODO 含义：jointly to minimize the TD-loss and maximize the Q_tot
        # Offline training and not target_net
        if not is_sample_action and not is_target_net and self.args.k_exchange > 0:
            raise NotImplementedError
            # Add noise in the form of 2-exchange neighborhoods
            random_shuffled_permutation_matrices = permutation_matrices.clone()  # [bs, n_agents, n_position, n_enemies]
            for k in range(self.args.k_exchange):
                # randomly choose two row indices.
                random_enemy_indices = np.random.choice(n_enemies, size=(2,), replace=False)
                # Inverse the order to achieve the 'swap' operation.
                swapped_enemy_indices = random_enemy_indices[
                                        ::-1].copy()  # ‘copy’ to ensure indice memory is contiguous
                # Swap the 2 rows  (TODO: Notice: in-place update, this is only used for online action sampling!)
                random_shuffled_permutation_matrices[:, :, random_enemy_indices] = random_shuffled_permutation_matrices[
                                                                                   :, :, swapped_enemy_indices]
            # generate random idx which use random permutation
            self.epsilon = self.schedule.eval(t_env)
            random_numbers = th.rand((bs, 1, 1, 1))  # [bs, 1, 1, 1]
            pick_random = (random_numbers < self.epsilon).long()
            permutation_matrices = pick_random * random_shuffled_permutation_matrices + (
                    1 - pick_random) * permutation_matrices

        # Inverse of permutation matrix -> permutation_matrix.transpose(-2, -1)
        inverse_permutation_matrices = permutation_matrices.transpose(-2, -1)  # [bs, n_agents, n_enemies, n_position]
        return permutation_matrices, inverse_permutation_matrices


class TopkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = [0.0, ]

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def get_topk_small(self):
        return self.data[0]

    def top_k(self):
        return heapq.nlargest(self.k, self.data)
        # return list(reversed([heapq.heappop(self.data) for _ in range(len(self.data))]))


# This multi-agent controller shares parameters between agents
class PermutationMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(PermutationMAC, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, is_sample_action=True, is_target_net=False, t_env=t_env)  # online sampling
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def _set_train_mode(self):
        self.agent.train()
        self.input_transformation_agent.train()

    def _set_evaluation_mode(self):
        self.agent.eval()
        self.input_transformation_agent.eval()

    def forward(self, ep_batch, t, is_sample_action, is_target_net, t_env):
        # (1) Remove the order;
        agent_inputs = self._build_inputs(ep_batch, t, is_sample_action, is_target_net, t_env)

        # (2) Forward;
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)  # [bs, n_agents, n_action]

        # (3) Recover the order;
        attack_qs = agent_outs[:, :, 6:].unsqueeze(-1)  # [bs, n_agent, n_position, 1]
        # [bs, n_agent, n_enemies, n_position] * [bs, n_agent, n_position, 1] = [bs, n_agent, n_enemies, 1]
        recovered_attack_qs = th.matmul(
            self.inverse_permutation_matrices, attack_qs
        ).squeeze(-1)  # [bs, n_agent, n_enemies]
        return th.cat([agent_outs[:, :, :6], recovered_attack_qs], dim=-1)

    def _is_time_to_permute(self, t, feats_t, feats_tm1, case=0):
        if case == 0:
            return t % self.args.permute_condition_value == 0
        else:
            if t == 0:
                self.heap = TopkHeap(k=self.args.permute_condition_value)
            # [bs, n_agent, fea_dim],  [bs]-> 0
            diff = th.abs(feats_t - feats_tm1).sum(dim=-1).sum(dim=-1).max(dim=0, keepdim=False)[0].squeeze().item()
            is_time = diff > self.heap.get_topk_small()
            self.heap.push(diff)
            # print("T={}, is_time={}, top-k={}".format(t, is_time, self.heap.top_k()))
            return is_time

    def _build_inputs(self, batch, t, is_sample_action, is_target_net, t_env):
        """
        # (1) Remove the order;
        :param batch:
        :param t:
        :param is_sample_action: only used in self.input_transformation_agent.forward
        :param is_target_net: only used in self.input_transformation_agent.forward
        :param t_env: only used in self.input_transformation_agent.forward
        :return:
        """
        bs = batch.batch_size
        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        move_feats_t, enemy_feats_t, ally_feats_t, own_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        _ally_feats_t = ally_feats_t.reshape(bs, self.n_agents, self.n_allies, -1)  # [bs, n_agents, n_allies,a_fea_dim]
        _enemy_feats_t = enemy_feats_t.reshape(bs, self.n_agents, self.n_enemies, -1)  # [bs,n_agents,n_enemies,fea_dim]

        if t == 0:
            self.feats_tm1 = enemy_feats_t
            # [bs, n_agents, n_enemies, n_position]
            matrix = th.eye(n=self.n_enemies, dtype=th.float32, device=raw_obs_t.device).unsqueeze(0).unsqueeze(
                0).expand(
                bs, self.n_agents, -1, -1
            )
            self.permutation_matrices, self.inverse_permutation_matrices = matrix, matrix
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%% compute assignment matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if self._is_time_to_permute(t, enemy_feats_t, self.feats_tm1, case=self.args.permute_condition):
            self.permutation_matrices, inverse_permutation_matrices = self.input_transformation_agent.forward(
                _ally_feats_t,
                _enemy_feats_t,
                is_sample_action=is_sample_action,
                is_target_net=is_target_net,
                t_env=t_env,
            )  # [bs, n_agents, n_position, n_enemies], [bs, n_agents, n_enemies, n_position]

            if self.args.detach_inverse_assignment:
                inverse_permutation_matrices = inverse_permutation_matrices.detach()

            self.inverse_permutation_matrices = inverse_permutation_matrices  # [bs, n_agents, n_enemies, n_position]
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%% compute assignment matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.feats_tm1 = enemy_feats_t  # feature of previous timestep

        # Re assign observations
        permuted_ally_feats_t = ally_feats_t
        # [bs, n_agents, n_position, n_enemies] * [bs, n_agents, n_enemies, fea_dim] -> [bs, n_agents, n_position * fea_dim]
        permuted_enemy_feats_t = th.matmul(self.permutation_matrices, _enemy_feats_t).view(bs, self.n_agents, -1)
        permuted_obs_t = th.cat([move_feats_t, permuted_enemy_feats_t, permuted_ally_feats_t, own_feats_t], dim=-1)

        inputs = []
        inputs.append(permuted_obs_t)  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                # Transform actions
                actions_onehot = batch["actions_onehot"][:, t - 1]  # [bs, n_agent, n_action]
                inputs.append(actions_onehot[:, :, :6])  # normal action [no-op, stop, north, south, east, west]
                attack_actions = actions_onehot[:, :, 6:].unsqueeze(-1)  # [bs, n_agent, n_enemies, 1]
                # [bs, n_agents, n_position, n_enemies] * [bs, n_agents, n_enemies, 1] = [bs, n_agents, n_position, 1] -> [bs, n_agents, n_position]
                permuted_attack_actions = th.matmul(self.permutation_matrices, attack_actions).squeeze(-1)
                inputs.append(permuted_attack_actions)
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        inputs = th.cat(inputs, dim=-1).contiguous()
        return inputs

    def prepare_permuted_q_for_training(self, batch, is_target_net):
        """
        enable large batch training to facilitate computation.
        :param batch:
        :return:
        """
        assert self.args.permute_condition == 0
        assert self.args.permute_condition_value == 1

        # (1) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Permute Observation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        bs, traj = batch.batch_size, batch.max_seq_length
        obs_component_dim, (ally_fea_dim, enemy_fea_dim) = self._get_obs_component_dim()
        move_feats, enemy_feats, ally_feats, own_feats = th.split(batch["obs"], obs_component_dim, dim=-1)
        # ally_feats = ally_feats.reshape(-1, self.n_agents, *ally_fea_dim)  # [bs*traj, n_agents, n_allies, a_fea_dim]
        enemy_feats = enemy_feats.reshape(-1, self.n_agents, *enemy_fea_dim)  # [bs*traj, n_agents, n_enemies, fea_dim]

        # %%%%%%%%%%%%% compute assignment matrix %%%%%%%%%%%%%
        permutation_matrices, inverse_permutation_matrices = self.input_transformation_agent.forward(
            ally_feats,
            enemy_feats,
            is_sample_action=False,
            is_target_net=is_target_net,
            t_env=-1,
        )  # [bs*traj, n_agents, n_position, n_enemies], [bs*traj, n_agents, n_enemies, n_position]
        if self.args.detach_inverse_assignment:
            inverse_permutation_matrices = inverse_permutation_matrices.detach()
        # %%%%%%%%%%%%% compute assignment matrix %%%%%%%%%%%%%

        # Re assign observations
        permuted_ally_feats = ally_feats
        # permuted_ally_feats = th.matmul(
        #     transposed_ally_assignments,  # [bs*traj, n_agents, n_position, n_allies]
        #     ally_feats  # [bs*traj, n_agents, n_allies, fea_dim]
        # ).view(bs, traj, self.n_agents, np.prod(ally_fea_dim))  # [bs, traj, n_agents, n_position * fea_dim]
        permuted_enemy_feats = th.matmul(
            permutation_matrices,  # [bs*traj, n_agents, n_position, n_enemies]
            enemy_feats  # [bs*traj, n_agents, n_enemies, fea_dim]
        ).view(bs, traj, self.n_agents, np.prod(enemy_fea_dim))  # [bs, traj, n_agents, n_position * fea_dim]

        reformat_obs = th.cat([move_feats, permuted_enemy_feats, permuted_ally_feats, own_feats], dim=-1)

        # Build Final Observations
        inputs = []
        inputs.append(reformat_obs)  # [bs, traj, n_agents, obs_dim]
        if self.args.obs_last_action:
            last_action = th.cat([th.zeros_like(batch["actions_onehot"][:, [0]]), batch["actions_onehot"][:, :-1]],
                                 dim=1)  # [bs, traj, n_agent, n_action]

            inputs.append(last_action[:, :, :, :6])  # normal action [no-op, stop, north, south, east, west]
            attack_actions = last_action[:, :, :, 6:].unsqueeze(-1)  # [bs, traj, n_agent, n_enemies, 1]
            permuted_attack_actions = th.matmul(
                permutation_matrices.view(bs, traj, self.n_agents, self.n_enemies, self.n_enemies),
                attack_actions
            ).squeeze(-1)  # [bs, traj, n_agent, n_enemies]
            inputs.append(permuted_attack_actions)

        if self.args.obs_agent_id:  # doesn't need to permute
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, traj, -1, -1))
        inputs = th.cat(inputs, dim=-1)  # [bs, traj, n_agent, final_obs_dim]

        # (2) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute Q-values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # Calculate estimated individual Q-Values
        mac_out = []
        self.init_hidden(batch.batch_size)
        for t in range(traj):
            iqs, self.hidden_states = self.agent(inputs[:, t].contiguous(), self.hidden_states)
            mac_out.append(iqs)  # [bs, n_agents, n_action]
        mac_out = th.stack(mac_out, dim=1)  # Concat over time, [bs, traj, n_agent, n_actions]

        # (3) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inverse Transformation of the attack actions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        attack_qs = mac_out[:, :, :, 6:].unsqueeze(-1)  # [bs, traj, n_agent, n_position, 1]
        # [bs, traj, n_agent, n_enemies, n_position] * [bs, traj, n_agent, n_position, 1] = [bs, traj, n_agent, n_enemies, 1]
        inversed_attack_qs = th.matmul(
            inverse_permutation_matrices.view(bs, traj, self.n_agents, self.n_enemies, self.n_enemies), attack_qs
        ).squeeze(-1)  # [bs, traj, n_agent, n_enemies]
        return th.cat([mac_out[:, :, :, :6], inversed_attack_qs], dim=-1)  # [bs, traj n_agent, n_actions]

    def parameters(self):
        return itertools.chain(
            self.agent.parameters(),
            self.input_transformation_agent.parameters()
        )

    def transform_parameters(self):
        return self.input_transformation_agent.parameters()

    def q_net_parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.input_transformation_agent.load_state_dict(other_mac.input_transformation_agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.input_transformation_agent.cuda()

    def cpu(self):
        self.agent.cpu()
        self.input_transformation_agent.cpu()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.input_transformation_agent.state_dict(), "{}/input_transformation_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.input_transformation_agent.load_state_dict(
            th.load("{}/input_transformation_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.input_transformation_agent = TransformationAgent(self.args)
        print("&&&&&&&&&&&&&&&&&&&&&&", self.args.agent, get_parameters_num(self.parameters()))

    # Add new func
    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            ally_feats_dim, enemy_feats_dim)
