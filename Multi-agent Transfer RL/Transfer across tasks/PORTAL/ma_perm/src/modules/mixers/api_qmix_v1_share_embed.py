import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class API_InputLayer(nn.Module):
    def __init__(self, args, output_dim):
        super(API_InputLayer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions

        self.embedding_enemy = nn.Linear(args.state_enemy_feats_size, output_dim)
        self.embedding_ally = nn.Linear(args.state_ally_feats_size, output_dim)
        if self.args.env_args["state_last_action"]:
            # self.embedding_action = nn.Linear(args.n_actions, output_dim)
            pass
        if self.args.env_args["state_timestep_number"]:
            self.embedding_timestep = nn.Linear(1, output_dim)

    def forward(self, states):
        state_components = th.split(states, self.args.state_component, dim=-1)
        # enemy
        enemy_features = state_components[1].reshape(-1, self.n_enemies,
                                                     self.args.state_enemy_feats_size)  # [bs * t, n_enemies, fea_dim]
        enemy_embedding = self.embedding_enemy(enemy_features).sum(dim=1, keepdim=False)

        # ally
        ally_features = state_components[0].reshape(-1, self.n_agents,
                                                    self.args.state_ally_feats_size)  # [bs * t, n_agents, fea_dim]
        ally_embedding = self.embedding_ally(ally_features).sum(dim=1, keepdim=False)

        output = enemy_embedding + ally_embedding

        if self.args.env_args["state_last_action"]:
            # n_agent_actions = state_components[2].reshape(-1, self.n_agents, self.n_actions)
            pass
        if self.args.env_args["state_timestep_number"]:
            timestep = state_components[-1].reshape(-1, 1)
            timestep_embedding = self.embedding_timestep(timestep)
            output = output + timestep_embedding
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
        self.hyper_w1 = nn.Sequential(API_InputLayer(args, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(API_InputLayer(args, self.embed_dim))

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(API_InputLayer(args, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(API_InputLayer(args, self.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, 1))

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim)  # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1)  # b * t, emb, 1
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2  # b * t, 1, 1

        return y.view(b, t, -1)

    def pos_func(self, x):
        return th.abs(x)
