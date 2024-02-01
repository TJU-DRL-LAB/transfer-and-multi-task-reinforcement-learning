#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ma_perm 
@File    ：api_controller.py
@Author  ：Hao Xiaotian
@Date    ：2021/12/9 22:32 
'''
import numpy as np
import torch as th

from .basic_controller import BasicMAC


class DataParallelAgent(th.nn.DataParallel):
    def init_hidden(self):
        # make hidden states on same device as model
        return self.module.init_hidden()

# This multi-agent controller shares parameters between agents
class APIMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(APIMAC, self).__init__(scheme, groups, args)
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
