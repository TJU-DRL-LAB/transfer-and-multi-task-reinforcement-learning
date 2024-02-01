import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class APIHyperInputLayer(nn.Module):
    def __init__(self, args, output_dim):
        super(APIHyperInputLayer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.output_dim = output_dim

        self.hyper_w_enemy = nn.Sequential(
            nn.Linear(args.state_enemy_feats_size, args.api_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.api_hyper_dim, args.state_enemy_feats_size * output_dim)
        )  # output shape: (enemy_feats_dim * output_dim)

        self.hyper_w_ally = nn.Sequential(
            nn.Linear(args.state_ally_feats_size, args.api_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.api_hyper_dim, args.state_ally_feats_size * output_dim)
        )  # output shape: (ally_feats_dim * output_dim)

        # if self.args.env_args["state_last_action"]:
        #     self.embedding_action = nn.Linear(args.n_actions, output_dim)
        #
        # if self.args.env_args["state_timestep_number"]:
        #     raise NotImplementedError
        #     # self.embedding_timestep = nn.Linear(1, output_dim)

    def forward(self, features):
        ally_features, enemy_features = features
        w_ally = self.hyper_w_ally(ally_features).view(-1, self.args.state_ally_feats_size, self.output_dim)
        # [bs * t * n_agents, 1, fea_dim]  * [bs * t * n_agents, fea_dim, output_dim]
        hidden_ally = th.matmul(ally_features.unsqueeze(1), w_ally).view(
            -1, self.n_agents, self.output_dim
        ).sum(dim=1, keepdim=False)  # [bs * t, output_dim]

        w_enemy = self.hyper_w_enemy(enemy_features).view(-1, self.args.state_enemy_feats_size, self.output_dim)
        # [bs * t * n_enemies, 1, fea_dim] * [bs * t * n_enemies, fea_dim, output_dim]
        hidden_enemy = th.matmul(enemy_features.unsqueeze(1), w_enemy).view(
            -1, self.n_enemies, self.output_dim
        ).sum(dim=1, keepdim=False)  # [bs * t, output_dim]

        output = hidden_enemy + hidden_ally
        # if self.args.env_args["state_last_action"]:
        #     n_agent_actions = state_components[2].reshape(-1, self.n_agents, self.n_actions)
        #     hidden_last_action = self.embedding_action(n_agent_actions).sum(dim=1, keepdim=False)  # [bs * t,output_dim]
        #     output = output + hidden_last_action
        #
        # if self.args.env_args["state_timestep_number"]:
        #     timestep = state_components[-1].reshape(-1, 1)
        return output


class APIHyperOutputLayer(nn.Module):
    def __init__(self, args, output_dim):
        super(APIHyperOutputLayer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_dim = output_dim

        self.hyper_w_ally = nn.Sequential(
            nn.Linear(1, args.api_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.api_hyper_dim, args.hypernet_embed * output_dim)
        )  # output shape: (ally_feats_dim * output_dim)

    def forward(self, hyper_w1_hidden, qs):
        """
        :param hyper_w1_hidden:
        :param qs:
        :return:
        """
        # [bs * t * n_agents, 1]

        # [bs * t, n_agents, hypernet_embed, output_dim] -> [bs * t, hypernet_embed, n_agents, output_dim]
        w_ally = self.hyper_w_ally(qs).view(
            -1, self.n_agents, self.args.hypernet_embed, self.output_dim
        ).transpose(1, 2).reshape(
            -1, self.args.hypernet_embed, self.n_agents * self.output_dim
        )  # [bs * t, hypernet_embed, n_agents * output_dim]

        # [bs * t, 1, hypernet_embed] * [bs * t, hypernet_embed, n_agents * output_dim]
        q_weight = th.matmul(hyper_w1_hidden.unsqueeze(1), w_ally).view(
            -1, self.n_agents, self.output_dim
        )  # [bs * t, n_agents, output_dim]
        return q_weight

# 92.609K
# 92.513K

class APIQMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(APIQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.abs = abs  # monotonicity constraint

        w_input_layer = APIHyperInputLayer(args, args.hypernet_embed)
        b_input_layer = APIHyperInputLayer(args, self.embed_dim)

        # hyper w1 b1
        self.hyper_w1_in = nn.Sequential(w_input_layer,
                                         nn.ReLU(inplace=True))
        self.hyper_w1_out = APIHyperOutputLayer(args, self.embed_dim)
        self.hyper_b1 = nn.Sequential(b_input_layer)

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(w_input_layer,
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(b_input_layer,
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, 1))

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
        features = (ally_features, enemy_features)

        # First layer
        w1 = self.hyper_w1_out(
            self.hyper_w1_in(features), qvals.view(-1, 1)
        ).view(-1, self.n_agents, self.embed_dim)  # b * t, n_agents, emb
        b1 = self.hyper_b1(features).view(-1, 1, self.embed_dim)

        # Second layer
        w2 = self.hyper_w2(features).view(-1, self.embed_dim, 1)  # b * t, emb, 1
        b2 = self.hyper_b2(features).view(-1, 1, 1)

        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2  # b * t, 1, 1

        return y.view(b, t, -1)

    def pos_func(self, x):
        return th.abs(x)
