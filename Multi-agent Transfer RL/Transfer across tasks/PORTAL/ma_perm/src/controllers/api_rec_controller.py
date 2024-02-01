#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch as th

from .basic_controller import BasicMAC


class DataParallelAgent(th.nn.DataParallel):
    def init_hidden(self):
        # make hidden states on same device as model
        return self.module.init_hidden()

# This multi-agent controller shares parameters between agents
class APIRECMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(APIRECMAC, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1

    # Add new func
    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        move_feats_t, enemy_feats_t, ally_feats_t, own_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        # merge move features and own features to simplify computation.
        context_feats = [move_feats_t, own_feats_t]  # [batch, agent_num, own_dim]
        own_context = th.cat(context_feats, dim=2).reshape(bs * self.n_agents, -1)  # [bs * n_agents, own_dim]

        embedding_indices = []
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1))
        if self.args.obs_last_action:
            # action-id indices, [bs, n_agents]
            if t == 0:
                embedding_indices.append(None)
            else:
                embedding_indices.append(batch["actions"][:, t - 1].squeeze(-1))

        return bs, own_context, enemy_feats_t, ally_feats_t, embedding_indices

    # add new func wjz
    def _get_obs_component_divide_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (8, 8), (7, 8), 4]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        feats_list = self.args.obs_component_divide  # feats_list是[4, (5, 8), (3, 8), (7, 8), 4]
        return_list = []
        for idx, feats in enumerate(feats_list):
            return_list.append(np.prod(feats))
        # 记录下各类enemy的数目
        enemy_nums = [n for n, _ in feats_list[1: -2]]
        return return_list, enemy_nums

    def _build_inputs_divide(self, batch, t):
        bs = batch.batch_size
        obs_component_dim, enemy_nums = self._get_obs_component_divide_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        split_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)

        move_feats_t= split_feats_t[0]
        enemy_feats_t = split_feats_t[1:-2]
        ally_feats_t= split_feats_t[-2]
        own_feats_t= split_feats_t[-1]
        enemy_feats_t_list = []
        for num, enemy_feats_ti in zip(enemy_nums, enemy_feats_t):
            enemy_feats_t_list.append(enemy_feats_ti.reshape(bs * self.n_agents * num,
                                                  -1))
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        # merge move features and own features to simplify computation.
        context_feats = [move_feats_t, own_feats_t]  # [batch, agent_num, own_dim]
        own_context = th.cat(context_feats, dim=2).reshape(bs * self.n_agents, -1)  # [bs * n_agents, own_dim]

        embedding_indices = []
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1))
        if self.args.obs_last_action:
            # action-id indices, [bs, n_agents]
            if t == 0:
                embedding_indices.append(None)
            else:
                embedding_indices.append(batch["actions"][:, t - 1].squeeze(-1))

        return bs, own_context, enemy_feats_t_list, ally_feats_t, embedding_indices

    def _get_input_shape(self, scheme):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        # 与环境交互时，不需要reconstruction data
        agent_outputs, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        # agent_inputs = self._build_inputs_divide(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_outs, self.hidden_states, rec_data, obs_embedding = self.agent(agent_inputs, self.hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), rec_data, obs_embedding

    def select_actions_with_feature(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, obs_feature, obs_feature_rnn = self.forward_with_feature(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions, obs_feature, obs_feature_rnn

    def forward_with_feature(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        # agent_inputs = self._build_inputs_divide(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        # todo 正常运行删除下面这行代码
        obs_feature, obs_feature_rnn = self.agent.forward_for_feature(agent_inputs, self.hidden_states)
        agent_outs, self.hidden_states, _, _ = self.agent(agent_inputs, self.hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), obs_feature, obs_feature_rnn

    def get_rec_r(self, obs_emb, action):
        rec_r = self.agent.predict_reward(obs_emb, action)
        return rec_r


    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
    # for rec1, don't reload decoder
    def load_models(self, path):
        model_loaded = th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        # 保存非decoder的weights
        state_dict = {k:v for k,v in model_loaded.items() if not (k.startswith("decoder") or k.startswith('r_pre'))}
        model_dict = self.agent.state_dict()
        model_dict.update(state_dict)
        self.agent.load_state_dict(model_dict)
        # self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))