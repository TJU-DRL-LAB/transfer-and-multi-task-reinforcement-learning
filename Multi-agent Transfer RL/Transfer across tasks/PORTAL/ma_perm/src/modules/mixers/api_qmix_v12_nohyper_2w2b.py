import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class APIQMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(APIQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.abs = abs  # monotonicity constraint

        # hyper w1 b1
        self.hyper_w1 = nn.Linear(self.n_agents, self.embed_dim, bias=True)

        # hyper w2 b2
        self.hyper_w2 = nn.Linear(self.embed_dim, 1, bias=True)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.view(b * t, 1, self.n_agents)  # (b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        state_components = th.split(states, self.args.state_component, dim=-1)
        # ally,  [bs * t * n_agents, fea_dim]
        ally_features = state_components[0].reshape(-1, self.args.state_ally_feats_size)
        # enemy, [bs * t * n_enemies, fea_dim]
        enemy_features = state_components[1].reshape(-1, self.args.state_enemy_feats_size)
        features = (ally_features, enemy_features)

        # First layer
        w1 = self.hyper_w1.weight  # [emb, n_agents]
        b1 = self.hyper_w1.bias

        # Second layer
        w2 = self.hyper_w2.weight  # [1, emb]
        b2 = self.hyper_w2.bias

        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, w1.t()) + b1)  # [b * t, 1, n_agents] * [n_agents, emb] = [b * t, 1, emb]
        y = th.matmul(hidden, w2.t()) + b2  # [b * t, 1, emb] * [emb, 1] = [b * t, 1, 1]

        return y.view(b, t, -1)

    def pos_func(self, x):
        return th.abs(x)
