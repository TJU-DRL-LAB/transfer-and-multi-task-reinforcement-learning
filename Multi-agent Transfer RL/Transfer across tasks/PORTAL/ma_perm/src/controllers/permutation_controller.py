import itertools

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.epsilon_schedules import DecayThenFlatSchedule
from modules.agents import REGISTRY as agent_REGISTRY
from .basic_controller import BasicMAC


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
        if self.args.assignment_use_RNN:
            raise NotImplementedError
            self.fc1 = nn.Linear(enemy_feats_dim, args.assignment_net_dim)
            self.rnn = nn.GRUCell(args.assignment_net_dim, args.assignment_net_dim)
            self.fc2 = nn.Linear(args.assignment_net_dim, n_enemies)
        else:
            # self.ally_assignment_policy = nn.Sequential(
            #     nn.Linear(ally_feats_dim, args.assignment_net_dim),
            #     nn.LeakyReLU(),
            #     nn.Linear(args.assignment_net_dim, n_allies),
            # )
            self.enemy_assignment_policy = nn.Sequential(
                nn.Linear(enemy_feats_dim, args.assignment_net_dim),
                # nn.LeakyReLU(),
                nn.ReLU(),
                nn.Linear(args.assignment_net_dim, n_enemies),
            )

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")

    def init_hidden(self, batch):
        if self.args.assignment_use_RNN:
            # [bs, n_agents, n_enemies, fea_dim]
            self.hidden_states = self.fc1.weight.new(1, 1, 1, self.args.assignment_net_dim).zero_().expand(
                batch,
                self.args.n_agents,
                self.args.n_enemies,
                -1
            )  # baev

    def _generate_action_prob(self, logit, batch_invalid_actions, test_mode):
        mask_inf = th.clamp(th.log(1 - batch_invalid_actions.float()), FLOAT_MIN,
                            FLOAT_MAX)  # 2-agent example: [[-inf, 0], [0, -inf]]
        logit_masked = logit + mask_inf
        if test_mode:
            return onehot_from_logits(logit_masked, dim=-1).detach()
        else:
            return F.gumbel_softmax(logit_masked, tau=self.args.softmax_tau, hard=True, dim=-1)

    def forward(self, ally_feats, enemy_feats, sample_action, test_mode, t_env):
        """
        :param ally_feats:  [bs, n_agents, n_allies, ally_fea_dim]
        :param enemy_feats:  [bs, n_agents, n_enemies, enemy_fea_dim]
        :return:  assignment: [bs, n_agents, n_enemies, n_position]
        """

        bs, n_agents, n_allies, ally_fea_dim = ally_feats.size()
        bs, n_agents, n_enemies, enemy_fea_dim = enemy_feats.size()

        if self.args.assignment_use_RNN:
            raise NotImplementedError
            x = F.relu(self.fc1(enemy_feats.view(-1, enemy_fea_dim)), inplace=True)
            h = self.rnn.forward(x, self.hidden_states.reshape(-1, self.args.assignment_net_dim))
            enemy_logits = self.fc2(h).view(bs, n_agents, n_enemies, n_enemies)  # [bs, n_agents, n_enemies, n_position]
            self.hidden_states = h  # update RNN hidden state.
        else:
            if self.args.transform_ally:
                ally_logits = self.ally_assignment_policy(ally_feats)  # [bs, n_agents, n_allies, n_ally_position]
            enemy_logits = self.enemy_assignment_policy(enemy_feats)  # [bs, n_agents, n_enemies, n_enemy_position]

        # the position order (learned order) is stable
        ally_assignments = None
        if self.args.transform_ally:
            # %%%%%%%%%%%%%%%%%%%%%%%%% Allies %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            _invalid_ally_mask = th.zeros(size=[bs, n_agents, n_allies], dtype=th.float,
                                          device=enemy_feats.device)  # [bs, n_agents, n_allies]  column order
            for position_idx in range(n_allies):  # over n_position
                logit = ally_logits[:, :, :, position_idx]  # [bs, n_agents, n_allies]  column order
                _ally_prob = self._generate_action_prob(logit, _invalid_ally_mask,
                                                        test_mode)  # [bs, n_agents, n_allies]
                _invalid_ally_mask = _invalid_ally_mask + _ally_prob.detach()
                ally_assignments.append(_ally_prob)
            ally_assignments = th.stack(ally_assignments, dim=-1)  # [bs, n_agents, n_allies, n_position]
            # %%%%%%%%%%%%%%%%%%%%%%%%% Allies %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # %%%%%%%%%%%%%%%%%%%%%%%%% Enemies %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        _invalid_enemy_mask = th.zeros(size=[bs, n_agents, n_enemies], dtype=th.float,
                                       device=enemy_feats.device)  # [bs, n_agents, n_enemies]  column order
        enemy_assignments = []
        for position_idx in range(n_enemies):  # over n_position, the position order (learned order) is stable
            logit = enemy_logits[:, :, :, position_idx]  # [bs, n_agents, n_enemies]  column order
            _enemy_prob = self._generate_action_prob(logit, _invalid_enemy_mask, test_mode)  # [bs, n_agents, n_enemies]
            _invalid_enemy_mask = _invalid_enemy_mask + _enemy_prob.detach()
            enemy_assignments.append(_enemy_prob)
        enemy_assignments = th.stack(enemy_assignments, dim=-1)  # [bs, n_agents, n_enemies, n_position]
        # %%%%%%%%%%%%%%%%%%%%%%%%% Enemies %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # exploration:
        if sample_action and not test_mode and self.args.k_exchange > 0:  # online sample action and not testing
            raise NotImplementedError
            # Add noise in the form of 2-exchange neighborhoods
            random_shuffled_ally_assignments = ally_assignments.clone()  # [bs, n_agents, n_allies, n_position]
            random_shuffled_enemy_assignments = enemy_assignments.clone()  # [bs, n_agents, n_enemies, n_position]

            for k in range(self.args.k_exchange):
                # randomly choose two row indices
                random_ally_indices = np.random.choice(n_allies, size=(2,), replace=False)
                swapped_ally_indices = random_ally_indices[::-1].copy()  # ‘copy’ to ensure indice memory is contiguous.
                # swap the 2 rows  (TODO: Notice: in-place update, this is only used for online action sampling!)
                random_shuffled_ally_assignments[:, :, random_ally_indices] = random_shuffled_ally_assignments[:, :,
                                                                              swapped_ally_indices]

                random_enemy_indices = np.random.choice(n_enemies, size=(2,), replace=False)
                swapped_enemy_indices = random_enemy_indices[::-1].copy()
                random_shuffled_enemy_assignments[:, :, random_enemy_indices] = random_shuffled_enemy_assignments[:, :,
                                                                                swapped_enemy_indices]

            # generate random idx which use random permutation
            self.epsilon = self.schedule.eval(t_env)
            random_numbers = th.rand((bs, n_agents, 1, 1))  # [bs, n_agents, 1, 1]
            pick_random = (random_numbers < self.epsilon).long()
            ally_assignments = pick_random * random_shuffled_ally_assignments + (1 - pick_random) * ally_assignments
            enemy_assignments = pick_random * random_shuffled_enemy_assignments + (1 - pick_random) * enemy_assignments

        # Inverse of assignment matrix inverse-> enemy_assignments.transpose(-2, -1) -> transpose -> enemy_assignments.transpose(-2, -1) = enemy_assignments
        transposed_inverse_enemy_assignments = enemy_assignments  # [bs, n_agents, n_position, n_enemies]
        return ally_assignments, enemy_assignments, transposed_inverse_enemy_assignments

    def get_unmasked_assignment_prob(self, ally_feats, enemy_feats):
        ally_prob = None
        if self.args.transform_ally:
            ally_logits = self.ally_assignment_policy(ally_feats)  # [bs, n_agents, n_allies, n_ally_position]
            ally_prob = F.softmax(ally_logits / self.args.softmax_tau, dim=-2)  # softmax
        enemy_logits = self.enemy_assignment_policy(enemy_feats)  # [bs, n_agents, n_enemies, n_enemy_position]
        enemy_prob = F.softmax(enemy_logits / self.args.softmax_tau, dim=-2)  # softmax
        return ally_prob, enemy_prob


class StateTransformationAgent(nn.Module):
    def __init__(self, args):
        super(StateTransformationAgent, self).__init__()
        self.args = args

        self.n_enemies, enemy_feats_dim = args.n_enemies, args.state_enemy_feats_size
        self.entity_assignment_policy = nn.Sequential(
            nn.Linear(enemy_feats_dim, args.assignment_net_dim),
            nn.LeakyReLU(),
            nn.Linear(args.assignment_net_dim, self.n_enemies),
        )
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")

    def _generate_action_prob(self, logit, batch_invalid_actions):
        mask_inf = th.clamp(th.log(1 - batch_invalid_actions.float()), FLOAT_MIN,
                            FLOAT_MAX)  # 2-agent example: [[-inf, 0], [0, -inf]]
        logit_masked = logit + mask_inf
        return F.gumbel_softmax(logit_masked, tau=self.args.softmax_tau, hard=True, dim=-1)

    def forward(self, ally_feats, enemy_feats):
        """
        :param ally_feats:  [bs, t, n_allies, ally_fea_dim]
        :param enemy_feats:  [bs, t, n_enemies, enemy_fea_dim]
        :return:  assignment: [bs, t, n_enemies, n_position]
        """
        logits = self.entity_assignment_policy(enemy_feats)  # [bs, t, n_enemies, n_position]
        bs, T = logits.shape[0], logits.shape[1]
        _invalid_enemy_mask = th.zeros(size=[bs, T, self.n_enemies], dtype=th.float,
                                       device=logits.device)  # [bs, n_enemies]  column order
        # the position order (learned order) is stable
        enemy_assignments = []
        for position_idx in range(self.n_enemies):  # over n_position
            logit = logits[:, :, :, position_idx]  # [bs, T, n_enemies]  column order
            _enemy_prob = self._generate_action_prob(logit, _invalid_enemy_mask)  # [bs, T, n_enemies]  column order
            _invalid_enemy_mask = _invalid_enemy_mask + _enemy_prob.detach()
            enemy_assignments.append(_enemy_prob)
        assignments = th.stack(enemy_assignments, dim=-1)  # [bs, T, n_enemies, n_position]
        return assignments

    def get_unmasked_assignment_prob(self, ally_feats, enemy_feats):
        enemy_logits = self.enemy_assignment_policy(enemy_feats)  # [bs, T, n_enemies, n_enemy_position]
        enemy_prob = F.softmax(enemy_logits / self.args.softmax_tau, dim=-2)  # softmax
        return enemy_prob


# This multi-agent controller shares parameters between agents
class PermutationMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(PermutationMAC, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1

        self.forward_times = 0
        self.permute_time = 0
        self.compute_q_time = 0

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        qvals = self.forward(ep_batch, t_ep, sample_action=True, test_mode=test_mode, use_stored=False,
                             t_env=t_env)  # online sampling
        chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def get_assignment(self, bs=slice(None)):
        return self.ally_assignments[bs], self.enemy_assignments[bs]

    def _set_train_mode(self):
        self.agent.train()
        self.input_transformation_agent.train()

    def _set_evaluation_mode(self):
        self.agent.eval()
        self.input_transformation_agent.eval()

    def forward(self, ep_batch, t, sample_action, test_mode, use_stored, t_env):
        if test_mode:
            self._set_evaluation_mode()
        agent_inputs = self._build_inputs(ep_batch, t, sample_action, test_mode, use_stored, t_env)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)  # [bs, n_agents, n_action]
        # TODO: Transform the attack actions
        attack_qs = agent_outs[:, :, 6:].unsqueeze(-1)  # [bs, n_agent, n_position, 1]
        # [bs, n_agent, n_position, n_enemies] -> transpose -> [bs, n_agent, n_enemies, n_position] * [bs, n_agent, n_position, 1] = [bs, n_agent, n_enemies, 1]
        reversed_attack_qs = th.matmul(
            self.transposed_inverse_enemy_assignments, attack_qs
        ).squeeze(-1)  # [bs, n_agent, n_enemies]
        return th.cat([agent_outs[:, :, :6], reversed_attack_qs], dim=-1)

    def parameters(self):
        # return self.agent.parameters()
        # if self.args.mixer == "qmix":
        #     return itertools.chain(
        #         self.agent.parameters(),
        #         self.input_transformation_agent.parameters(),
        #         self.state_transformation_agent.parameters(),
        #     )
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
        # if self.args.mixer == "qmix":
        #     self.state_transformation_agent.load_state_dict(other_mac.state_transformation_agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.input_transformation_agent.cuda()
        # if self.args.mixer == "qmix":
        #     self.state_transformation_agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.input_transformation_agent.state_dict(), "{}/input_transformation_agent.th".format(path))
        # if self.args.mixer == "qmix":
        #     th.save(self.state_transformation_agent.state_dict(), "{}/state_transformation_agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.input_transformation_agent.load_state_dict(
            th.load("{}/input_transformation_agent.th".format(path), map_location=lambda storage, loc: storage))
        # if self.args.mixer == "qmix":
        #     self.state_transformation_agent.load_state_dict(
        #         th.load("{}/state_transformation_agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.input_transformation_agent = TransformationAgent(self.args)
        # if self.args.mixer == "qmix":
        #     self.state_transformation_agent = StateTransformationAgent(self.args)

    # Add new func
    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            ally_feats_dim, enemy_feats_dim)

    def _build_inputs(self, batch, t, sample_action, test_mode, use_stored, t_env):
        bs = batch.batch_size
        if t == 0:  # init
            self.input_transformation_agent.init_hidden(bs)  # TODO: init hidden state for RNN

        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        move_feats_t, enemy_feats_t, ally_feats_t, own_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        _ally_feats_t = ally_feats_t.reshape(bs, self.n_agents, self.n_allies, -1)  # [bs, n_agents, n_allies,a_fea_dim]
        _enemy_feats_t = enemy_feats_t.reshape(bs, self.n_agents, self.n_enemies, -1)  # [bs,n_agents,n_enemies,fea_dim]

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%% compute assignment matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if t % self.args.permute_condition_value == 0:
            if use_stored:  # Training
                raise NotImplementedError
                ally_assignments, enemy_assignments = batch["ally_assignment"][:, t], batch["enemy_assignment"][:, t]
                ally_prob, enemy_prob = self.input_transformation_agent.get_unmasked_assignment_prob(_ally_feats_t,
                                                                                                     _enemy_feats_t)
                ally_assignments = (ally_assignments - ally_prob.detach()) + ally_prob
                enemy_assignments = (enemy_assignments - enemy_prob.detach()) + enemy_prob
                transposed_inverse_enemy_assignments = enemy_assignments  # [bs, n_agents, n_enemies, n_position]
            else:  # Online action selection
                ally_assignments, enemy_assignments, transposed_inverse_enemy_assignments = self.input_transformation_agent.forward(
                    _ally_feats_t,
                    _enemy_feats_t,
                    sample_action,  # if sample_action and not test_mode, then add \epsilon-Greedy exploration.
                    # test_mode=True if self.args.random_projection else test_mode,
                    test_mode=True if self.args.random_projection else False,
                    t_env=t_env,
                )  # [bs, n_agents, n_allies, n_position],  [bs, n_agents, n_enemies, n_position], [bs, n_agents, n_position, n_enemies]
                self.ally_assignments = ally_assignments
                self.enemy_assignments = enemy_assignments

            if self.args.detach_inverse_assignment:
                transposed_inverse_enemy_assignments = transposed_inverse_enemy_assignments.detach()
            if self.args.transform_ally:
                self.transposed_ally_assignments = th.transpose(ally_assignments, -1,
                                                                -2)  # [bs, n_agents, n_pos, n_allies]
            self.transposed_enemy_assignments = th.transpose(enemy_assignments, -1, -2)  # [bs,n_agents,n_pos,n_enemies]
            self.transposed_inverse_enemy_assignments = transposed_inverse_enemy_assignments  # [bs, n_agents, n_enemies, n_position]
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%% compute assignment matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # Re assign observations
        if self.args.transform_ally:
            re_assigned_ally_feats_t = th.matmul(
                self.transposed_ally_assignments,  # [bs, n_agents, n_position, n_allies]
                _ally_feats_t  # [bs, n_agents, n_allies, fea_dim]
            ).view(bs, self.n_agents, -1)  # [bs, n_agents, n_position * fea_dim]
        else:
            re_assigned_ally_feats_t = ally_feats_t

        permuted_enemy_feats_t = th.matmul(
            self.transposed_enemy_assignments,  # [bs, n_agents, n_position, n_enemies]
            _enemy_feats_t  # [bs, n_agents, n_enemies, fea_dim]
        ).view(bs, self.n_agents, -1)  # [bs, n_agents, n_position * fea_dim]
        reformat_obs_t = th.cat([move_feats_t, permuted_enemy_feats_t, re_assigned_ally_feats_t, own_feats_t],
                                dim=-1)

        inputs = []
        inputs.append(reformat_obs_t)  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                # Transform actions
                actions_onehot = batch["actions_onehot"][:, t - 1]  # [bs, n_agent, n_action]
                inputs.append(actions_onehot[:, :, :6])  # normal action [no-op, stop, north, south, east, west]
                attack_actions = actions_onehot[:, :, 6:].unsqueeze(-1)  # [bs, n_agent, n_enemies, 1]
                permuted_attack_actions = th.matmul(self.transposed_enemy_assignments,  # [bs,1,n_position,n_enemies]
                                                    attack_actions).squeeze(-1)  # [bs, n_agents, n_position]
                inputs.append(permuted_attack_actions)
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        inputs = th.cat(inputs, dim=-1).contiguous()
        return inputs

    def prepare_permuted_q_for_training(self, batch, use_stored, is_target=False):
        """
        enable large batch training to facilitate computation.
        :param batch:
        :param use_stored:
        :return:
        """
        raise NotImplementedError
        # start_time = time.time()
        self.forward_times += 1
        # (1) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Permute Observation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        bs, traj = batch.batch_size, batch.max_seq_length
        obs_component_dim, (ally_fea_dim, enemy_fea_dim) = self._get_obs_component_dim()
        move_feats, enemy_feats, ally_feats, own_feats = th.split(batch["obs"], obs_component_dim, dim=-1)
        ally_feats = ally_feats.reshape(-1, self.n_agents, *ally_fea_dim)  # [bs*traj, n_agents, n_allies, a_fea_dim]
        enemy_feats = enemy_feats.reshape(-1, self.n_agents, *enemy_fea_dim)  # [bs*traj, n_agents, n_enemies, fea_dim]
        if use_stored:  # Training
            ally_assignments, enemy_assignments = batch["ally_assignment"], batch["enemy_assignment"]
            ally_prob, enemy_prob = self.input_transformation_agent.get_unmasked_assignment_prob(ally_feats,
                                                                                                 enemy_feats)
            ally_assignments = (ally_assignments.reshape(bs * traj, self.n_agents, self.n_allies,
                                                         self.n_allies) - ally_prob.detach()) + ally_prob
            enemy_assignments = (enemy_assignments.reshape(bs * traj, self.n_agents, self.n_enemies,
                                                           self.n_enemies) - enemy_prob.detach()) + enemy_prob
            transposed_inverse_enemy_assignments = enemy_assignments  # [bs, traj,n_agents,n_enemies,n_position]
        else:
            # On-policy generation
            ally_assignments, enemy_assignments, transposed_inverse_enemy_assignments = self.input_transformation_agent.forward(
                ally_feats,
                enemy_feats,
                False,  # if sample_action and not test_mode, then add \epsilon-Greedy exploration.
                test_mode=False,
                t_env=None,
            )  # [bs*traj, n_agents, n_allies, n_position],  [bs*traj, n_agents, n_enemies, n_position], [bs*traj, n_agents, n_position, n_enemies]

        if self.args.detach_inverse_assignment:
            transposed_inverse_enemy_assignments = transposed_inverse_enemy_assignments.detach()

        transposed_ally_assignments = th.transpose(ally_assignments, -1, -2)  # [bs*traj, n_agents, n_pos, n_allies]
        transposed_enemy_assignments = th.transpose(enemy_assignments, -1, -2)  # [bs*traj, n_agents, n_pos, n_enemies]
        transposed_inverse_enemy_assignments = transposed_inverse_enemy_assignments.view(
            bs, traj, self.n_agents, self.n_enemies, self.n_enemies
        )  # [bs, traj, n_agents, n_position, n_enemies]

        # Re assign observations
        permuted_ally_feats = th.matmul(
            transposed_ally_assignments,  # [bs*traj, n_agents, n_position, n_allies]
            ally_feats  # [bs*traj, n_agents, n_allies, fea_dim]
        ).view(bs, traj, self.n_agents, np.prod(ally_fea_dim))  # [bs, traj, n_agents, n_position * fea_dim]
        permuted_enemy_feats = th.matmul(
            transposed_enemy_assignments,  # [bs*traj, n_agents, n_position, n_enemies]
            enemy_feats  # [bs*traj, n_agents, n_enemies, fea_dim]
        ).view(bs, traj, self.n_agents, np.prod(enemy_fea_dim))  # [bs, traj, n_agents, n_position * fea_dim]

        reformat_obs = th.cat([move_feats, permuted_enemy_feats, permuted_ally_feats, own_feats],
                              dim=-1)

        # Build Final Observations
        inputs = []
        inputs.append(reformat_obs)  # [bs, traj, n_agents, obs_dim]
        if self.args.obs_last_action:
            last_action = th.cat([th.zeros_like(batch["actions_onehot"][:, [0]]), batch["actions_onehot"][:, :-1]],
                                 dim=1)  # [bs, traj, n_agent, n_action]
            inputs.append(last_action[:, :, :, :6])  # normal action [no-op, stop, north, south, east, west]

            attack_actions = last_action[:, :, :, 6:].unsqueeze(-1)  # [bs, traj, n_agent, n_enemies, 1]
            permuted_attack_actions = th.matmul(
                transposed_enemy_assignments.view(bs, traj, self.n_agents, self.n_enemies, self.n_enemies),
                attack_actions
            ).squeeze(-1)  # [bs, traj, n_agent, n_enemies]
            inputs.append(permuted_attack_actions)

        if self.args.obs_agent_id:  # doesn't need to permute
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, traj, -1, -1))
        # inputs = th.cat([x.reshape(bs, traj, self.n_agents, -1) for x in inputs], dim=-1)  # [bs, traj, n_agent, final_obs_dim]
        inputs = th.cat(inputs, dim=-1)  # [bs, traj, n_agent, final_obs_dim]

        # if not is_target:
        #     permuted_finished_time = time.time()
        #     self.permute_time += (permuted_finished_time - start_time)
        #     print("Permute avg costs {} seconds".format(self.permute_time / self.forward_times))

        # (2) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Compute Q-values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Calculate estimated individual Q-Values
        mac_out = []
        self.init_hidden(batch.batch_size)
        for t in range(traj):
            iqs, self.hidden_states = self.agent(inputs[:, t].contiguous(),
                                                 self.hidden_states)  # [bs, n_agents, n_action]
            # TODO: Inverse Transformation of the attack actions
            attack_qs = iqs[:, :, 6:].unsqueeze(-1)  # [bs, n_agent, n_position, 1]
            # [bs, n_agent, n_position, n_enemies] -> transpose -> [bs, n_agent, n_enemies, n_position] * [bs, n_agent, n_position, 1] = [bs, n_agent, n_enemies, 1]
            inversed_attack_qs = th.matmul(
                transposed_inverse_enemy_assignments[:, t], attack_qs
            ).squeeze(-1)  # [bs, n_agent, n_enemies]
            mac_out.append(th.cat([iqs[:, :, :6], inversed_attack_qs], dim=-1))  # [bs, n_agent, n_actions]

        mac_out = th.stack(mac_out, dim=1)  # Concat over time, [bs, traj, n_agent, n_actions]

        # if not is_target:
        #     cal_q_time = time.time()
        #     self.compute_q_time += (cal_q_time - permuted_finished_time)
        #     print("Compute q avg costs {} seconds".format(self.compute_q_time / self.forward_times))

        return mac_out

    def transform_states(self, states):
        """
        TODO: for Mixing net
        :param states: [bs, t, state_dim]
        :return:
        """
        return states

        state_components = th.split(states, self.args.state_component, dim=-1)
        # enemy
        enemy_features = state_components[1].reshape(*states.shape[:2], self.n_enemies,
                                                     self.args.state_enemy_feats_size)  # [bs, t, n_enemies, fea_dim]

        # Get assignment matrix
        assignments = self.state_transformation_agent.forward(None, enemy_features)
        transposed_assignment = assignments.transpose(-1, -2)  # [bs, T, n_position, n_enemies]

        # Permute state
        permuted_enemy_feats = th.matmul(transposed_assignment, enemy_features)  # [bs, t, n_position, fea_dim]
        permuted_enemy_feats = permuted_enemy_feats.reshape(*states.shape[:2],
                                                            self.n_enemies * self.args.state_enemy_feats_size)

        reformat_state = [state_components[0], permuted_enemy_feats]
        # state_last_action
        if self.args.env_args["state_last_action"]:
            last_action = state_components[2].reshape(
                *states.shape[:2], self.n_agents, self.n_actions
            )  # [bs, t, n_agents, n_action]

            attack_actions = last_action[:, :, :, 6:].unsqueeze(-1)  # [bs, t, n_agent, n_enemies, 1]
            permuted_attack_actions = th.matmul(
                transposed_assignment.unsqueeze(dim=2),  # [bs, t, 1, n_position, n_enemies]
                attack_actions
            ).squeeze(-1)  # [bs, t, n_agents, n_position]
            # normal action [no-op, stop, north, south, east, west]
            permuted_last_action = th.cat([last_action[:, :, :, :6], permuted_attack_actions], dim=-1)
            permuted_last_action = permuted_last_action.reshape(*states.shape[:2],
                                                                self.n_agents * self.n_actions)  # [bs, t, n_agents * n_action]
            reformat_state.append(permuted_last_action)
        if self.args.env_args["state_timestep_number"]:
            reformat_state.append(state_components[-1])
        reformat_state = th.cat(reformat_state, dim=-1)
        return reformat_state
