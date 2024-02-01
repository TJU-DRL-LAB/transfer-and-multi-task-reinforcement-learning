import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class APIHypernet(nn.Module):
    def __init__(self, args):
        super(APIHypernet, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.embed_dim = args.mixing_embed_dim  # hidden dim of the Main net
        self.hyper_embed_dim = args.hypernet_embed  # hidden dim of the hyper net

        # TODO: 先不要 bias-1

        self.hyper_w_enemy = nn.Sequential(
            nn.Linear(args.state_enemy_feats_size, args.api_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.api_hyper_dim, args.state_enemy_feats_size * self.hyper_embed_dim)
        )  # output shape: (enemy_feats_dim * hyper_embed_dim)

        self.hyper_w_ally = nn.Sequential(
            nn.Linear(args.state_ally_feats_size, args.api_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.api_hyper_dim, args.state_ally_feats_size * self.hyper_embed_dim)
        )  # output shape: (ally_feats_dim * hyper_embed_dim)

        # if self.args.env_args["state_last_action"]:
        #     self.embedding_action = nn.Linear(args.n_actions, hyper_embed_dim)
        #
        # if self.args.env_args["state_timestep_number"]:
        #     raise NotImplementedError
        #     # self.embedding_timestep = nn.Linear(1, hyper_embed_dim)

        self.hyper_q_outdim = self.hyper_embed_dim * self.embed_dim
        self.hyper_w_q = nn.Sequential(
            nn.Linear(1, args.api_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.api_hyper_dim, self.hyper_q_outdim)
        )

        self.hyper_main_w2b2 = nn.Linear(self.hyper_embed_dim, self.embed_dim + 1)

    def forward(self, ally_features, enemy_features, qvals):
        """
        :param ally_features:
        :param enemy_features:
        :param qvals:  [b, t, n_agents]
        :return:
        """
        # (1) Generate hidden for hypernet.
        w_ally = self.hyper_w_ally(ally_features).view(-1, self.args.state_ally_feats_size, self.hyper_embed_dim)
        # [bs * t * n_agents, 1, fea_dim]  * [bs * t * n_agents, fea_dim, hyper_embed_dim]
        hidden_ally = th.matmul(ally_features.unsqueeze(1), w_ally).view(
            -1, self.n_agents, self.hyper_embed_dim
        ).sum(dim=1, keepdim=False)  # [bs * t, hyper_embed_dim]

        w_enemy = self.hyper_w_enemy(enemy_features).view(-1, self.args.state_enemy_feats_size, self.hyper_embed_dim)
        # [bs * t * n_enemies, 1, fea_dim] * [bs * t * n_enemies, fea_dim, hyper_embed_dim]
        hidden_enemy = th.matmul(enemy_features.unsqueeze(1), w_enemy).view(
            -1, self.n_enemies, self.hyper_embed_dim
        ).sum(dim=1, keepdim=False)  # [bs * t, hyper_embed_dim]

        hyper_hidden = (hidden_enemy + hidden_ally)  # [bs * t, hyper_embed_dim]
        # if self.args.env_args["state_last_action"]:
        #     n_agent_actions = state_components[2].reshape(-1, self.n_agents, self.n_actions)
        #     hidden_last_action = self.embedding_action(n_agent_actions).sum(dim=1, keepdim=False)  # [bs * t,hyper_embed_dim]
        #     output = output + hidden_last_action
        # if self.args.env_args["state_timestep_number"]:
        #     timestep = state_components[-1].reshape(-1, 1)

        # (2) Generate weights and bias
        normalized_qvals = F.softmax(qvals.detach(), dim=-1).view(-1, 1)
        hyper_w2 = self.hyper_w_q(normalized_qvals).view(
            -1, self.n_agents, self.hyper_embed_dim, self.embed_dim
        )  # [b*t, n_agents, hyper_embed_dim, embed_dim]
        hyper_w2 = hyper_w2.transpose(1, 2).reshape(
            -1, self.hyper_embed_dim, self.n_agents * self.embed_dim
        )  # [b*t, hyper_embed_dim, n_agents, embed_dim] -> [b*t, hyper_embed_dim, n_agents * embed_dim]

        main_w1 = th.matmul(hyper_hidden.unsqueeze(dim=1), hyper_w2).view(
            -1, self.n_agents, self.embed_dim
        )  # [b*t, n_agents, embed_dim]

        main_w2b2 = self.hyper_main_w2b2(hyper_hidden)  # [bs * t, embed_dim + 1]
        main_w2 = main_w2b2[:, :-1].unsqueeze(dim=-1)  # [b*t, embed_dim, 1]
        main_b2 = main_w2b2[:, -1:].unsqueeze(dim=-1)  # [b*t, 1, 1]

        return main_w1, main_w2, main_b2


class APIQMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(APIQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.abs = abs  # monotonicity constraint

        # hyper w1 b1
        self.hyper_ws = APIHypernet(args)


    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()

        states = states.reshape(-1, self.state_dim)
        state_components = th.split(states, self.args.state_component, dim=-1)
        # ally,  [bs * t * n_agents, fea_dim]
        ally_features = state_components[0].reshape(-1, self.args.state_ally_feats_size)
        # enemy, [bs * t * n_enemies, fea_dim]
        enemy_features = state_components[1].reshape(-1, self.args.state_enemy_feats_size)

        main_w1, main_w2, main_b2 = self.hyper_ws.forward(ally_features, enemy_features, qvals)


        qvals = qvals.view(b * t, 1, self.n_agents)  # (b * t, 1, self.n_agents)

        if self.abs:
            main_w1 = self.pos_func(main_w1)
            main_w2 = self.pos_func(main_w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, main_w1))  # [b * t, 1, emb]
        y = th.matmul(hidden, main_w2) + main_b2  # [b * t, 1, 1]

        return y.view(b, t, -1)

    def pos_func(self, x):
        return th.abs(x)
