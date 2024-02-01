import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DeepSetRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DeepSetRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = self.n_agents - 1
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions

        # [4 + 1, (6, 5), (4, 5)], take 5m_vs_6m for example
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]

        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.args.rnn_hidden_dim)

        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.args.rnn_hidden_dim)

        # Own features
        self.fc1_own = nn.Linear(self.own_feats_dim, args.rnn_hidden_dim, bias=True)  # only one bias is OK

        # Ally features
        self.fc1_ally = nn.Linear(self.ally_feats_dim, args.rnn_hidden_dim, bias=False)  # only one bias is OK

        # Enemy features
        self.fc1_enemy = nn.Linear(self.enemy_feats_dim, args.rnn_hidden_dim, bias=False)  # only one bias is OK

        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # print(self.fc1_ally.weight.data.mean(), self.fc1_ally.weight.data.var())
        # print(self.fc1_enemy.weight.data.mean(), self.fc1_enemy.weight.data.var())

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state=None):
        # [bs * n_agents, mv_fea_dim+own_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t.reshape(-1, self.own_feats_dim))
        # (2) ID embeddings
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.args.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.args.rnn_hidden_dim)

        # (3) Enemy feature
        embedding_enemies = self.fc1_enemy(enemy_feats_t).view(
            bs * self.n_agents, self.n_enemies, self.args.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, rnn_hidden_dim]
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]

        # (4) Ally features
        embedding_allies = self.fc1_ally(ally_feats_t).view(
            bs * self.n_agents, self.n_allies, self.args.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]
        aggregated_embedding = embedding_own + embedding_enemies + embedding_allies  # [bs * n_agents, rnn_hidden_dim]

        x = F.relu(aggregated_embedding, inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q.view(bs, self.n_agents, -1), h.view(bs, self.n_agents, -1)