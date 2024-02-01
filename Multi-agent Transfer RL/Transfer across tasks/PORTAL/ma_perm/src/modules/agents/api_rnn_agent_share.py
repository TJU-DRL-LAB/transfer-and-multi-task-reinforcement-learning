import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


def kaiming_uniform_(tensor, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with th.no_grad():
        return tensor.uniform_(-bound, bound)


class API_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(API_RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = self.n_agents - 1
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.rnn_hidden_dim = args.rnn_hidden_dim

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]

        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # Unique Features (do not need hyper net)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK

        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_fc1_w_enemy = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, args.api_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.api_hyper_dim, (self.enemy_feats_dim + 1) * self.rnn_hidden_dim + 1)
        )  # output shape: (enemy_feats_dim * self.rnn_hidden_dim + self.rnn_hidden_dim + 1)

        if self.args.map_type == "MMM":
            assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
            self.hyper_fc1_w_ally = nn.Sequential(
                nn.Linear(self.ally_feats_dim, args.api_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.api_hyper_dim, (self.ally_feats_dim + 1) * self.rnn_hidden_dim + 1)
            )  # output shape: ally_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1, for 'rescue actions'
        else:
            self.hyper_fc1_w_ally = nn.Sequential(
                nn.Linear(self.ally_feats_dim, args.api_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.api_hyper_dim, self.ally_feats_dim * self.rnn_hidden_dim)
            )  # output shape: ally_feats_dim * rnn_hidden_dim

        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)

        self.fc2_normal_actions = nn.Linear(self.rnn_hidden_dim, 6)  # (no_op, stop, up, down, right, left)

        # Reset parameters for hypernets
        # self._reset_hypernet_parameters()

    def _reset_hypernet_parameters(self):
        gain = 2 ** (-0.5)
        for m in self.hyper_fc1_w_enemy.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight.data, gain=gain)
                m.bias.data.fill_(0.)
        for m in self.hyper_fc1_w_ally.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight.data)
                kaiming_uniform_(m.weight.data, gain=gain)
                m.bias.data.fill_(0.)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]

        # (2) ID embeddings
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0]
            # [bs * n_agents, rnn_hidden_dim]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                # [bs * n_agents, rnn_hidden_dim]
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.rnn_hidden_dim)

        # (3) Enemy feature  (enemy_feats_dim * self.rnn_hidden_dim + self.rnn_hidden_dim + 1)
        hyper_enemy_out = self.hyper_fc1_w_enemy(enemy_feats_t)
        fc1_w_enemy = hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1)].reshape(
            -1, self.enemy_feats_dim, self.rnn_hidden_dim
        )  # [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim]
        # [bs * n_agents * n_enemies, 1, enemy_fea_dim] * [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), fc1_w_enemy).view(
            bs * self.n_agents, self.n_enemies, self.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, rnn_hidden_dim]
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]

        # (4) Ally features
        hyper_ally_out = self.hyper_fc1_w_ally(ally_feats_t)
        if self.args.map_type == "MMM":
            fc1_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim + 1)].reshape(
                -1, self.ally_feats_dim, self.rnn_hidden_dim
            )  # [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim]
        else:
            fc1_w_ally = hyper_ally_out.view(-1, self.ally_feats_dim, self.rnn_hidden_dim)
        # [bs * n_agents * n_allies, 1, ally_fea_dim] * [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim] = [bs * n_agents * n_allies, 1, rnn_hidden_dim]
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), fc1_w_ally).view(
            bs * self.n_agents, self.n_allies, self.rnn_hidden_dim
        )  # [bs * n_agents, n_allies, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]
        # Final embedding
        embedding = embedding_own + embedding_enemies + embedding_allies  # [bs * n_agents, rnn_hidden_dim]

        # print(fc1_w_enemy.data.mean(), fc1_w_enemy.data.var())
        # print(fc1_w_ally.data.mean(), fc1_w_ally.data.var())

        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]

        # Q-values of normal actions
        q_normal = self.fc2_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]

        # Q-values of attack actions
        fc2_w_attack = hyper_enemy_out[:, -(self.rnn_hidden_dim + 1): -1].reshape(
            bs * self.n_agents, self.n_enemies, self.rnn_hidden_dim
        ).transpose(-2, -1)  # [bs * n_agents * n_enemies, rnn_hidden_dim] -> [bs * n_agents, rnn_hidden_dim, n_enemies]
        fc2_b_attack = hyper_enemy_out[:, -1:].reshape(bs * self.n_agents, self.n_enemies)
        q_attack = (th.matmul(hh.unsqueeze(1), fc2_w_attack).squeeze(1) + fc2_b_attack).view(bs, self.n_agents,
                                                                                             -1)  # [bs, n_agents, n_enemies]

        # TODO: 'rescue' actions for map_type == "MMM"
        if self.args.map_type == "MMM":
            fc2_w_rescue = hyper_ally_out[:, -(self.rnn_hidden_dim + 1): -1].reshape(
                bs * self.n_agents, self.n_allies, self.rnn_hidden_dim
            ).transpose(-2, -1)  # [bs*n_agents*n_allies, rnn_hidden_dim] -> [bs * n_agents, rnn_hidden_dim, n_allies]
            fc2_b_rescue = hyper_ally_out[:, -1:].reshape(bs * self.n_agents, self.n_allies)
            q_rescue = (th.matmul(hh.unsqueeze(1), fc2_w_rescue).squeeze(1) + fc2_b_rescue).view(bs, self.n_agents,
                                                                                                 -1)  # [bs, n_agents, n_allies]
            # TODO: Currently, for MMM and MMM2 map, only the last agents are medivacs.
            modified_q_attack_of_medivac = th.cat([q_rescue[:, -1:, :], q_attack[:, -1:, self.n_allies:]], dim=-1)
            q_attack = th.cat([q_attack[:, :-1], modified_q_attack_of_medivac], dim=1)

        # Concat 2 types of Q-values
        q = th.cat((q_normal, q_attack), dim=-1)  # [bs, n_agents, 6 + n_enemies]

        return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1)  # [bs, n_agents, 6 + n_enemies]
