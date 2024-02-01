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
            # nn.ReLU(inplace=True),
            nn.PReLU(num_parameters=args.api_hyper_dim),
            nn.Linear(args.api_hyper_dim, args.state_enemy_feats_size * output_dim)
        )  # output shape: (enemy_feats_dim * output_dim)

        self.hyper_w_ally = nn.Sequential(
            nn.Linear(args.state_ally_feats_size, args.api_hyper_dim),
            # nn.ReLU(inplace=True),
            nn.PReLU(num_parameters=args.api_hyper_dim),
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

class APIQMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(APIQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.abs = abs  # monotonicity constraint

        # hyper w1 b1
        self.hyper_w1 = nn.Linear(self.n_agents, self.embed_dim, bias=False)

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(APIHyperInputLayer(args, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))

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
        w1 = self.hyper_w1.weight  # [emb, n_agents]

        # Second layer
        w2 = self.hyper_w2(features).view(-1, self.embed_dim, 1)  # b * t, emb, 1

        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, w1.t()))  # [b * t, 1, n_agents] * [n_agents, emb] = [b * t, 1, emb]
        y = th.matmul(hidden, w2)  # b * t, 1, 1

        return y.view(b, t, -1)

    def pos_func(self, x):
        return th.abs(x)
