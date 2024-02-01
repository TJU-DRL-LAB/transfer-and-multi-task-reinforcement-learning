import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter


def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)
    std = gain / math.sqrt(fan)
    bound_w = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    bound_b = 1 / math.sqrt(fan)
    with th.no_grad():
        tensor_w.data.uniform_(-bound_w, bound_w)
        if tensor_b is not None:
            tensor_b.data.uniform_(-bound_b, bound_b)


class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)


class API_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(API_RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = self.n_agents - 1
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.api_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.unit_type_bits = args.map_unit_type_bits
        self.obs_component_divide = args.obs_component_divide
        self.enemy_divide = self.obs_component_divide[1:-2]
        self.ally_divide = self.obs_component_divide[-2]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # Unique Features (do not need hyper net)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK

        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_enemys = nn.ModuleList()
        for i in range(len(self.enemy_divide)):
            self.hyper_enemys.append(nn.Sequential(
                nn.Linear(self.enemy_feats_dim, args.api_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.api_hyper_dim, ((self.enemy_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
            ))  # output shape: (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1)
        if self.args.map_type == "MMM":
            assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
            self.hyper_ally = nn.Sequential(
                nn.Linear(self.ally_feats_dim, args.api_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.api_hyper_dim, ((self.ally_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
            )  # output shape: ally_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1, for 'rescue actions'
            self.unify_output_heads_rescue = Merger(self.n_heads, 1)
        else:
            self.hyper_allies = nn.ModuleList()
            for i in range(1):
                self.hyper_allies.append(nn.Sequential(
                    nn.Linear(self.ally_feats_dim, args.api_hyper_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.api_hyper_dim, self.ally_feats_dim * self.rnn_hidden_dim * self.n_heads)
                ))  # output shape: ally_feats_dim * rnn_hidden_dim)
        self.unify_input_heads = Merger(self.n_heads, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2_normal_actions = nn.Linear(self.rnn_hidden_dim, 6)  # (no_op, stop, up, down, right, left)
        self.unify_output_heads = Merger(self.n_heads, 1)

        # Reset parameters for hypernets
        self._reset_hypernet_parameters(init_type="xavier")
        # self._reset_hypernet_parameters(init_type="kaiming")

    def _reset_hypernet_parameters(self, init_type='kaiming'):
        gain = 2 ** (-0.5)
        # %%%%%%%%%%%%%%%%%%%%%% Hypernet-based API input layer %%%%%%%%%%%%%%%%%%%%
        for i in range(len(self.enemy_divide)):
            for m in self.hyper_enemys[i].modules():
                if isinstance(m, nn.Linear):
                    if init_type == "kaiming":
                        kaiming_uniform_(m.weight, m.bias, gain=gain)
                    else:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.)
        for i in range(1):
            for m in self.hyper_allies[i].modules():
                if isinstance(m, nn.Linear):
                    if init_type == "kaiming":
                        kaiming_uniform_(m.weight, m.bias, gain=gain)
                    else:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()

    def divide_feats1(self, feats_t):
        feats_list = []
        idxs = [[] for _ in range(self.unit_type_bits)]
        for idx, x in enumerate(feats_t):
            has_category = False
            for i in range(self.unit_type_bits):
                if x[-self.unit_type_bits+i].item() == 1:
                    idxs[i].append(idx)
                    has_category = True
            if not has_category:
                idxs[0].append(idx)
        # get divided tensor list
        for i in range(self.unit_type_bits):
            if len(idxs[i]) > 0:
                feats_list.append(torch.index_select(feats_t, dim=0, index=torch.tensor(idxs[i])))
            else:
                feats_list.append(torch.tensor([]))
        return feats_list, idxs

    def forward(self, inputs, hidden_state):
        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, enemy_feats_list, ally_feats_list, embedding_indices = inputs

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

        # (3) Enemy feature  (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1)
        embedding_enemies_list = []
        hyper_enemy_out_list = []
        if not isinstance(enemy_feats_list, list):
            enemy_feats_list = [enemy_feats_list]
        for i, enemy_feats_t in enumerate(enemy_feats_list):
            hyper_enemy_out = self.hyper_enemys[i](enemy_feats_t)
            hyper_enemy_out_list.append(hyper_enemy_out)
            fc1_w_enemy = hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
                -1, self.enemy_feats_dim, self.rnn_hidden_dim * self.n_heads
            )  # [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim]
            # [bs * n_agents * n_enemies, 1, enemy_fea_dim] * [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]
            embedding_enemies_i = th.matmul(enemy_feats_t.unsqueeze(1), fc1_w_enemy).view(
                bs * self.n_agents, -1, self.n_heads, self.rnn_hidden_dim
            )  # [bs * n_agents, n_enemies, rnn_hidden_dim * head]
            embedding_enemies_list.append(embedding_enemies_i)
        embedding_enemies = torch.cat(embedding_enemies_list, dim=1)
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, rnn_hidden_dim]

        # (4) Ally features
        if not isinstance(ally_feats_list, list):
            ally_feats_list = [ally_feats_list]
        embedding_allies_list = []
        for i, ally_feats_t in enumerate(ally_feats_list):
            hyper_ally_out = self.hyper_allies[i](ally_feats_t)
            if self.args.map_type == "MMM": # todo 异质
                fc1_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
                    -1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads
                )  # [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim * head]
            else:
                fc1_w_ally = hyper_ally_out.view(-1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads)
            # [bs * n_agents * n_allies, 1, ally_fea_dim] * [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim] = [bs * n_agents * n_allies, 1, rnn_hidden_dim]
            embedding_allies_i = th.matmul(ally_feats_t.unsqueeze(1), fc1_w_ally).view(
                bs * self.n_agents, -1, self.n_heads, self.rnn_hidden_dim
            )  # [bs * n_agents, n_enemies, rnn_hidden_dim * head]
            embedding_allies_list.append(embedding_allies_i)
        embedding_allies = torch.cat(embedding_allies_list, dim=1)
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, rnn_hidden_dim]

        # Final embedding
        embedding = embedding_own + self.unify_input_heads(
            embedding_enemies + embedding_allies
        )  # [bs * n_agents, head, rnn_hidden_dim]

        # print(fc1_w_enemy.data.mean(), fc1_w_enemy.data.var())
        # print(fc1_w_ally.data.mean(), fc1_w_ally.data.var())

        x = F.relu(embedding, inplace=True)
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]

        # Q-values of normal actions
        q_normal = self.fc2_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]

        # Q-values of attack actions
        q_attacks_type_list = []
        for num_of_type, hyper_enemy_out in zip(self.enemy_divide, hyper_enemy_out_list):
            fc2_w_attack = hyper_enemy_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
                bs * self.n_agents, num_of_type[0] * self.n_heads, self.rnn_hidden_dim
            ).transpose(-2, -1)  # [bs*n_agents,n_enemies*head,rnn_hidden_dim]->[bs*n_agents,rnn_hidden_dim,n_enemies*head]
            fc2_b_attack = hyper_enemy_out[:, -self.n_heads:].reshape(bs * self.n_agents, num_of_type[0] * self.n_heads)
            # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_enemies*head] -> [bs*n_agents, 1, n_enemies*head]
            q_attacks_type_list_i = (th.matmul(hh.unsqueeze(1), fc2_w_attack).squeeze(1) + fc2_b_attack).\
                view(bs * self.n_agents, num_of_type[0], self.n_heads)
            q_attacks_type_list.append(q_attacks_type_list_i)
        q_attacks = torch.cat(q_attacks_type_list, dim=1).view(
            bs * self.n_agents * self.n_enemies, self.n_heads, 1
        )  # [bs * n_agents, n_enemies*head] -> [bs * n_agents * n_enemies, head, 1]

        # Merge multiple heads into one.
        q_attack = self.unify_output_heads(q_attacks).view(  # [bs * n_agents * n_enemies, 1]
            bs, self.n_agents, self.n_enemies
        )  # [bs, n_agents, n_enemies]

        # %%%%%%%%%%%%%%% 'rescue' actions for map_type == "MMM" %%%%%%%%%%%%%%%
        if self.args.map_type == "MMM":
            # todo divide for WWW
            fc2_w_rescue = hyper_ally_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
                bs * self.n_agents, self.n_allies * self.n_heads, self.rnn_hidden_dim
            ).transpose(-2, -1)  # [bs*n_agents, n_allies*head, rnn_hidden] -> [bs*n_agents, rnn_hidden, n_allies*head]
            fc2_b_rescue = hyper_ally_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_allies * self.n_heads)
            # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_allies*head] -> [bs*n_agents, 1, n_allies*head]
            q_rescues = (th.matmul(hh.unsqueeze(1), fc2_w_rescue).squeeze(1) + fc2_b_rescue).view(
                bs * self.n_agents * self.n_allies, self.n_heads, 1
            )  # [bs * n_agents, n_allies*head] -> [bs * n_agents * n_allies, head, 1]
            # Merge multiple heads into one.
            q_rescue = self.unify_output_heads_rescue(q_rescues).view(  # [bs * n_agents * n_allies, 1]
                bs, self.n_agents, self.n_allies
            )  # [bs, n_agents, n_allies]

            # TODO: Currently, for MMM and MMM2 map, only the last agents are medivacs (n_enemies > n_agents).
            # For the reason that medivac is the last indexed agent, so the rescue action idx -> [0, n_allies-1]
            right_padding = th.ones_like(q_attack[:, -1:, self.n_allies:], requires_grad=False) * -9999999
            modified_q_attack_of_medivac = th.cat([q_rescue[:, -1:, :], right_padding], dim=-1)
            q_attack = th.cat([q_attack[:, :-1], modified_q_attack_of_medivac], dim=1)

        # Concat 2 types of Q-values
        q = th.cat((q_normal, q_attack), dim=-1)  # [bs, n_agents, 6 + n_enemies]
        return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1)  # [bs, n_agents, 6 + n_enemies]

if __name__ == '__main__':
    pass