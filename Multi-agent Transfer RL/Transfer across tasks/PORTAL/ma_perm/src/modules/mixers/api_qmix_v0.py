import numpy as np
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


class API_InputLayer(nn.Module):
    def __init__(self, args, output_dim):
        super(API_InputLayer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.output_dim = output_dim

        if args.shared_hyper_input_layer_num == 1:
            self.embedding_ally = nn.Linear(args.state_ally_feats_size, output_dim, bias=False)
            self.embedding_enemy = nn.Linear(args.state_enemy_feats_size, output_dim, bias=False)
        else:
            assert args.shared_hyper_input_layer_num == 2
            self.embedding_ally = nn.Sequential(
                nn.Linear(args.state_ally_feats_size, args.api_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.api_hyper_dim, output_dim, bias=False)
            )
            self.embedding_enemy = nn.Sequential(
                nn.Linear(args.state_enemy_feats_size, args.api_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.api_hyper_dim, output_dim, bias=False)
            )
        self.bias = Parameter(th.Tensor(output_dim), requires_grad=True)
        fan_in = args.state_enemy_feats_size * self.n_enemies + args.state_ally_feats_size * self.n_agents
        bound_b = 1 / math.sqrt(fan_in)
        self.bias.data.uniform_(-bound_b, bound_b)

        if self.args.env_args["state_last_action"]:
            # we do not use action_idx.
            # self.embedding_action = nn.Linear(args.n_actions, output_dim)
            pass
        if self.args.env_args["state_timestep_number"]:
            self.embedding_timestep = nn.Linear(1, output_dim, bias=False)

    def forward(self, state_components):
        # ally
        ally_features = state_components[0]  # [bs * t * n_agents, fea_dim]
        ally_embedding = self.embedding_ally(ally_features).view(
            -1, self.n_agents, self.output_dim
        ).sum(dim=1, keepdim=False)

        # enemy
        enemy_features = state_components[1]  # [bs * t * n_enemies, fea_dim]
        enemy_embedding = self.embedding_enemy(enemy_features).view(
            -1, self.n_enemies, self.output_dim
        ).sum(dim=1, keepdim=False)

        output = enemy_embedding + ally_embedding + self.bias

        if self.args.env_args["state_last_action"]:
            # n_agent_actions = state_components[2]
            pass
        if self.args.env_args["state_timestep_number"]:
            timestep = state_components[-1]
            timestep_embedding = self.embedding_timestep(timestep)
            output = output + timestep_embedding
        return output


class API_HyperW1_OutputLayer(nn.Module):
    def __init__(self, args):
        super(API_HyperW1_OutputLayer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.hyper_embed_dim = args.hypernet_embed  # hidden dim of the hyper net
        self.embed_dim = args.mixing_embed_dim  # hidden dim of the Main net

        self.hyper_w_ally_and_b = nn.Sequential(
            nn.Linear(args.state_ally_feats_size, args.api_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.api_hyper_dim, self.hyper_embed_dim * self.embed_dim + self.embed_dim)
        )

    def forward(self, ally_features, hyper_w1_hidden):
        """
            :param ally_features: [bs * t * n_agents, fea_dim]
            :param hyper_w1_hidden:  # [bs * t, hyper_embed_dim]
            :return:
        """
        hyper_out = self.hyper_w_ally_and_b(ally_features).view(
            -1, self.n_agents, self.hyper_embed_dim + 1, self.embed_dim
        )  # [b * t, n_agents, hyper_embed_dim+1, embed_dim]

        w_ally = hyper_out[:, :, :-1]  # ]b * t, n_agents, hyper_embed_dim, embed_dim]
        b_ally = hyper_out[:, :, -1:]  # [b * t, n_agents, 1, embed_dim]

        w_ally = w_ally.transpose(1, 2).reshape(
            -1, self.hyper_embed_dim, self.n_agents * self.embed_dim
        )  # [b*t, hyper_embed_dim, n_agents, embed_dim] -> [b*t, hyper_embed_dim, n_agents * embed_dim]
        b_ally = b_ally.reshape(
            -1, 1, self.n_agents * self.embed_dim
        )  # [bs * t, 1, n_agents * embed_dim]

        # [bs * t, 1, hyper_embed_dim] * [bs * t, hyper_embed_dim, n_agents * embed_dim] -> [bs * t, 1, n_agents * embed_dim]
        main_w1 = th.matmul(hyper_w1_hidden.unsqueeze(dim=1), w_ally) + b_ally
        return main_w1


class HyperW1(nn.Module):
    def __init__(self, args):
        super(HyperW1, self).__init__()
        self.hyper_w1_input = nn.Sequential(
            API_InputLayer(args, args.hypernet_embed),
            nn.ReLU(inplace=True)
        )
        self.hyper_w1_output = API_HyperW1_OutputLayer(args)

    def forward(self, state_components):
        hyper_w1_hidden = self.hyper_w1_input(state_components)
        main_w1 = self.hyper_w1_output.forward(state_components[0], hyper_w1_hidden)
        return main_w1


class APIQMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(APIQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.embed_dim = args.mixing_embed_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.abs = abs  # monotonicity constraint
        assert self.abs

        # hyper w1 b1
        self.hyper_w1 = HyperW1(args)
        self.hyper_b1 = nn.Sequential(API_InputLayer(args, self.embed_dim))

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(API_InputLayer(args, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(API_InputLayer(args, self.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, 1))

        self._init_parameters()

    def _init_parameters(self):
        # xavier-init main networks before training
        gain = 12 ** (-0.5)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, m.bias, gain=gain)
                # m.bias.data.fill_(0.)

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)
        state_components = list(th.split(states, self.args.state_component, dim=-1))
        # ally feature
        state_components[0] = state_components[0].reshape(-1,
                                                          self.args.state_ally_feats_size)  # [bs * t * n_agents, fea_dim]
        # enemy feature
        state_components[1] = state_components[1].reshape(-1,
                                                          self.args.state_enemy_feats_size)  # [bs * t * n_enemies, fea_dim]
        if self.args.env_args["state_last_action"]:
            # state_components[2] = state_components[2].reshape(-1, self.n_agents, self.n_actions)
            pass
        if self.args.env_args["state_timestep_number"]:
            state_components[-1] = state_components[-1].reshape(-1, 1)

        # First layer
        w1 = self.hyper_w1(state_components).view(-1, self.n_agents, self.embed_dim)  # b * t, n_agents, emb
        b1 = self.hyper_b1(state_components).view(-1, 1, self.embed_dim)

        # Second layer
        w2 = self.hyper_w2(state_components).view(-1, self.embed_dim, 1)  # b * t, emb, 1
        b2 = self.hyper_b2(state_components).view(-1, 1, 1)

        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # print(w1.mean(), w1.var())
        # print(w2.mean(), w2.var())

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2  # b * t, 1, 1

        return y.view(b, t, -1)

    def pos_func(self, x):
        return th.abs(x)
