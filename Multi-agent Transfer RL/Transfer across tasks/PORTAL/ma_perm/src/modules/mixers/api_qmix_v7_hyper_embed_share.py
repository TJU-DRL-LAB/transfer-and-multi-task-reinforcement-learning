import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class APIHyperLayer1(nn.Module):
    def __init__(self, args):
        super(APIHyperLayer1, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_and_b_ally = nn.Sequential(
            nn.Linear(args.state_ally_feats_size, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, 2 * self.embed_dim)
        )  # output shape: (1 * embed_dim + embed_dim)

        # self.hyper_w_and_b_enemy = nn.Sequential(
        #     nn.Linear(args.state_enemy_feats_size, args.hypernet_embed),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(args.hypernet_embed, (args.state_enemy_feats_size + 1) * self.embed_dim)
        # )  # output shape: (enemy_feats_dim * embed_dim + embed_dim)

        self.hyper_w_and_b_enemy = nn.Sequential(
            nn.Linear(args.state_enemy_feats_size, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, 2 * self.embed_dim)
        )  # output shape: (embed_dim + embed_dim)

    def forward(self, qvals, ally_features, enemy_features):
        """
        qvals: [bs * t, 1, n_agents]
        ally_features: [bs * t * n_agents, fea_dim]
        enemy_features: [bs * t * n_enemies, fea_dim]
        """

        # (1) hyper w1 b1 of ally
        # [bs * t * n_agents, 1 + 1, embed_dim]
        w_and_b_ally = self.hyper_w_and_b_ally(ally_features).view(-1, 2, self.embed_dim)
        w_ally = w_and_b_ally[:, :-1].reshape(
            -1, self.n_agents, self.embed_dim
        )  # [bs * t, n_agents, embed_dim]
        b_ally = w_and_b_ally[:, -1].reshape(
            -1, self.n_agents, self.embed_dim
        ).sum(dim=1, keepdim=False)  # [bs * t, embed_dim]
        # [bs * t, 1, n_agents] *  [bs * t, n_agents, embed_dim] = [bs * t, 1, embed_dim]
        hidden_ally = th.matmul(qvals, th.abs(w_ally)).squeeze(dim=1) + b_ally  # [bs * t, embed_dim]

        # (2) hyper w1 b1 of enemy
        # # [bs * t * n_enemies, enemy_feats_dim + 1, embed_dim]
        # w_and_b_enemy = self.hyper_w_and_b_enemy(enemy_features).view(
        #     -1, self.args.state_enemy_feats_size + 1, self.embed_dim
        # )
        # w_enemy = w_and_b_enemy[:, :-1]
        # b_enemy = w_and_b_enemy[:, -1].reshape(
        #     -1, self.n_enemies, self.embed_dim
        # ).sum(dim=1, keepdim=False)  # [bs * t, embed_dim]
        # # [bs * t * n_enemies, 1, fea_dim] * [bs * t * n_enemies, fea_dim, embed_dim]
        # hidden_enemy = th.matmul(enemy_features.unsqueeze(1), w_enemy).view(
        #     -1, self.n_enemies, self.embed_dim
        # ).sum(dim=1, keepdim=False) + b_enemy  # [bs * t, embed_dim]

        w_and_b_enemy = self.hyper_w_and_b_enemy(enemy_features).view(-1, 2, self.embed_dim)
        w_enemy = w_and_b_enemy[:, :-1].reshape(
            -1, self.n_enemies, self.embed_dim
        ).sum(dim=1, keepdim=True).expand(-1, self.n_agents, -1)  # [bs * t, n_agents, embed_dim]
        b_enemy = w_and_b_enemy[:, -1].reshape(
            -1, self.n_enemies, self.embed_dim
        ).sum(dim=1, keepdim=False)  # [bs * t, embed_dim]
        # [bs * t, 1, n_agents] *  [bs * t, n_agents, embed_dim] = [bs * t, 1, embed_dim]
        hidden_enemy= th.matmul(qvals, th.abs(w_enemy)).squeeze(dim=1) + b_enemy  # [bs * t, embed_dim]


        output = hidden_ally + hidden_enemy
        return output


class APIHyperLayer2(nn.Module):
    def __init__(self, args):
        super(APIHyperLayer2, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_and_b_ally = nn.Sequential(
            nn.Linear(args.state_ally_feats_size, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.embed_dim * 1 + 1)
        )  # output shape: (embed_dim * 1)

        self.hyper_w_and_b_enemy = nn.Sequential(
            nn.Linear(args.state_enemy_feats_size, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.embed_dim * 1 + 1)
        )  # output shape: (embed_dim * 1)

    def forward(self, hidden, ally_features, enemy_features):
        """
        hidden: [bs * t, embed_dim]
        ally_features: [bs * t * n_agents, fea_dim]
        enemy_features: [bs * t * n_enemies, fea_dim]
        """

        # (1) hyper w1 b1 of ally
        # [bs * t * n_agents, embed_dim + 1, 1]
        w_and_b_ally = self.hyper_w_and_b_ally(ally_features).view(-1, self.embed_dim * 1 + 1, 1)
        w_ally = w_and_b_ally[:, :-1].reshape(
            -1, self.n_agents, self.embed_dim, 1
        ).sum(dim=1, keepdim=False)  # [bs * t * n_agents, embed_dim, 1] -> [bs * t, embed_dim, 1]
        b_ally = w_and_b_ally[:, -1].reshape(
            -1, self.n_agents, 1
        ).sum(dim=1, keepdim=False)  # [bs * t, 1]
        # [bs * t, 1, embed_dim] *  [bs * t, embed_dim, 1] = [bs * t, 1, 1]
        output_ally = th.matmul(hidden.unsqueeze(1), th.abs(w_ally)).squeeze(dim=1) + b_ally  # [bs * t, 1]

        # (2) hyper w1 b1 of enemy
        # [bs * t * n_enemies, embed_dim + 1, 1]
        w_and_b_enemy = self.hyper_w_and_b_enemy(enemy_features).view(-1, self.embed_dim * 1 + 1, 1)
        w_enemy = w_and_b_enemy[:, :-1].reshape(
            -1, self.n_enemies, self.embed_dim, 1
        ).sum(dim=1, keepdim=False)   # [bs * t, embed_dim, 1]
        b_enemy = w_and_b_enemy[:, -1].reshape(
            -1, self.n_enemies, 1
        ).sum(dim=1, keepdim=False)  # [bs * t, 1]
        # [bs * t, 1, embed_dim] * [bs * t, embed_dim, 1] = [bs * t, 1, 1]
        output_enemy = th.matmul(hidden.unsqueeze(1), th.abs(w_enemy)).squeeze(dim=1) + b_enemy  # [bs * t, 1]
        return output_ally + output_enemy


class APIQMixer(nn.Module):
    def __init__(self, args):
        super(APIQMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        # Layer-1
        self.layer_1 = APIHyperLayer1(args)

        # Layer-2
        self.layer_2 = APIHyperLayer2(args)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        state_components = th.split(states, self.args.state_component, dim=-1)
        # ally,  [bs * t * n_agents, fea_dim]
        ally_features = state_components[0].reshape(-1, self.args.state_ally_feats_size)

        # enemy, [bs * t * n_enemies, fea_dim]
        enemy_features = state_components[1].reshape(-1, self.args.state_enemy_feats_size)

        # First layer
        hidden = F.elu(self.layer_1(qvals, ally_features, enemy_features))
        # Second layer
        y = self.layer_2(hidden, ally_features, enemy_features)
        return y.view(b, t, -1)
