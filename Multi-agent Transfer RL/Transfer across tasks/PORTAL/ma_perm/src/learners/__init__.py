from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .ppo_learner import PPOLearner
from .lica_learner import LICALearner
from .nq_learner_release import NQLearner
# from .nq_learner_split import NQLearner as NQLearnerSplit
from .policy_gradient_v2 import PGLearner_v2
from .max_q_learner import MAXQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .offpg_learner import OffPGLearner
from .fmac_learner import FMACLearner
# from .permutation_q_learner_largebatch_release import PermutationQLearner
from .permutation_q_learner_release import PermutationQLearner
# from .permutation_q_learner_release_1lossfunc import PermutationQLearner
# from .permutation_q_learner_faster import PermutationQLearner as PermutationQLearnerFaster
# from .nq_learner_data_augmentation import NQLearnerDataAugmentation

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["lica_learner"] = LICALearner
REGISTRY["nq_learner"] = NQLearner
# REGISTRY["nq_learner_split"] = NQLearnerSplit
REGISTRY["policy_gradient_v2"] = PGLearner_v2
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["offpg_learner"] = OffPGLearner
REGISTRY["fmac_learner"] = FMACLearner
REGISTRY["permutation_q_learner"] = PermutationQLearner
# REGISTRY["permutation_q_learner_faster"] = PermutationQLearnerFaster
# REGISTRY["q_learner_data_augmentation"] = NQLearnerDataAugmentation