REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent
from .noisy_agents import NoisyRNNAgent
from .api_rnn_agent_share import API_RNNAgent
from .api_rnn_agent import API_RNNAgent as API_RNNAgent_noshare
from .api_rnn_agent_3layer import API_RNNAgent as API_RNNAgent_3layer
from .api_rnn_agent_residual import API_RNNAgent as API_RNNAgent_residual
from .api_rnn_agent_densenet import API_RNNAgent as API_RNNAgent_densenet
from .api_rnn_agent_densenet_share import API_RNNAgent as API_RNNAgent_densenet_share
from .api_rnn_agent_residual_share import API_RNNAgent as API_RNNAgent_residual_share
from .api_rnn_agent_multihead import API_RNNAgent as API_RNNAgent_multihead
from .api_rnn_agent_multihead_share import API_RNNAgent as API_RNNAgent_multihead_share
from .api_rnn_agent_multihead_share_relation import API_RNNAgent as API_RNNAgent_multihead_share_relation
from .api_rnn_agent_multihead_share_relation_v23 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v23
from .api_rnn_agent_multihead_share_relation_v31 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v31
from .api_rnn_agent_multihead_share_relation_v311 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v311
from .api_rnn_agent_multihead_share_relation_v32 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v32
from .api_rnn_agent_multihead_share_relation_v321 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v321
from .api_rnn_agent_multihead_share_relation_v33 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v33
from .api_rnn_agent_multihead_share_relation_v33_dividehyper import API_RNNAgent as API_RNNAgent_multihead_share_relation_v33_dividehyper
from .api_rnn_agent_multihead_share_relation_v0 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v0
from .api_rnn_agent_multihead_share_relation_v4 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v4
from .api_rnn_agent_multihead_share_relation_v51 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v51
from .api_rnn_agent_multihead_share_relation_v52 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v52
from .api_rnn_agent_multihead_share_relation_v53 import API_RNNAgent as API_RNNAgent_multihead_share_relation_v53
from .api_rnn_agent_multihead_share_dividehyper import API_RNNAgent as API_RNNAgent_multihead_share_dividehyper
from .deepset_rnn_agent import DeepSetRNNAgent
from .deepset_hyper_rnn_agent import DeepSetHyperRNNAgent
from .asn_rnn_agent import AsnRNNAgent
from .gnn_rnn_agent import GnnRNNAgent
from .dyan_attackunit_rnn import DyanRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent
REGISTRY["api_rnn"] = API_RNNAgent
REGISTRY["api_rnn_noshare"] = API_RNNAgent_noshare
REGISTRY["api_rnn_3layer"] = API_RNNAgent_3layer
REGISTRY["api_rnn_residual"] = API_RNNAgent_residual
REGISTRY["api_rnn_densenet"] = API_RNNAgent_densenet
REGISTRY["api_rnn_densenet_share"] = API_RNNAgent_densenet_share
REGISTRY["api_rnn_residual_share"] = API_RNNAgent_residual_share
REGISTRY["api_rnn_multihead"] = API_RNNAgent_multihead
REGISTRY["api_rnn_multihead_share"] = API_RNNAgent_multihead_share
REGISTRY["api_rnn_multihead_share_relation"] = API_RNNAgent_multihead_share_relation
REGISTRY["api_rnn_multihead_share_relation_v23"] = API_RNNAgent_multihead_share_relation_v23
REGISTRY["api_rnn_multihead_share_relation_v31"] = API_RNNAgent_multihead_share_relation_v31
REGISTRY["api_rnn_multihead_share_relation_v311"] = API_RNNAgent_multihead_share_relation_v311
REGISTRY["api_rnn_multihead_share_relation_v32"] = API_RNNAgent_multihead_share_relation_v32
REGISTRY["api_rnn_multihead_share_relation_v321"] = API_RNNAgent_multihead_share_relation_v321
REGISTRY["api_rnn_multihead_share_relation_v33"] = API_RNNAgent_multihead_share_relation_v33
REGISTRY["api_rnn_multihead_share_relation_v33_dividehyper"] = API_RNNAgent_multihead_share_relation_v33_dividehyper
REGISTRY["api_rnn_multihead_share_relation_v4"] = API_RNNAgent_multihead_share_relation_v4
REGISTRY["api_rnn_multihead_share_relation_v51"] = API_RNNAgent_multihead_share_relation_v51
REGISTRY["api_rnn_multihead_share_relation_v52"] = API_RNNAgent_multihead_share_relation_v52
REGISTRY["api_rnn_multihead_share_relation_v53"] = API_RNNAgent_multihead_share_relation_v53
REGISTRY["api_rnn_multihead_share_relation_v0"] = API_RNNAgent_multihead_share_relation_v0
REGISTRY["api_rnn_multihead_share_dividehyper"] = API_RNNAgent_multihead_share_dividehyper
REGISTRY["api_rnn_multihead_share_rec"] = API_RNNAgent_multihead_share_rec
REGISTRY["api_rnn_multihead_share_rec1"] = API_RNNAgent_multihead_share_rec1
REGISTRY["deepset_rnn"] = DeepSetRNNAgent
REGISTRY["deepset_hyper_rnn"] = DeepSetHyperRNNAgent
from .updet_agent import UPDeT
REGISTRY["updet_agent"] = UPDeT
from .updet_agent_hxt import UPDeT as UPDeTHXT
REGISTRY["updet_agent_hxt"] = UPDeTHXT
REGISTRY["asn_rnn"] = AsnRNNAgent
REGISTRY["gnn_rnn"] = GnnRNNAgent
REGISTRY["dyan_attackunit_rnn"] = DyanRNNAgent


