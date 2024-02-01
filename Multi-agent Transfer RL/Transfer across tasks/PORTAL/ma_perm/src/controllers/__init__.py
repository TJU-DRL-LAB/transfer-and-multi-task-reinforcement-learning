REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC
from .permutation_controller_release import PermutationMAC
from .api_controller import APIMAC
from .api_rec_controller import APIRECMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC
REGISTRY["permutation_mac"] = PermutationMAC
REGISTRY["api_mac"] = APIMAC
# for reconstruction
REGISTRY["api_rec_mac"] = APIRECMAC

from .updet_controller import UPDETController
REGISTRY["updet_mac"] = UPDETController
