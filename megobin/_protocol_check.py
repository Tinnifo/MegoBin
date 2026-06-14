"""Static Protocol-conformance assertions.

Each assignment forces mypy to verify, structurally, that the concrete
class satisfies the Protocol. A class that drops or renames a Protocol
method fails mypy here — the CI gate catches it instead of a runtime
``AttributeError`` 20 minutes into training.

This module is import-only for the type checker. It is not executed
in production code paths and has no runtime effect beyond the imports
themselves.
"""

from megobin.binners.base import Binner
from megobin.binners.semibin_2 import DBSCANEnsembleBinner
from megobin.data.base import PairSampler
from megobin.data.uncertain_gen_sampler import UncertainGenPairSampler
from megobin.data.semibin2_sampler import SemiBin2PairSampler
from megobin.encoders.base import Encoder
from megobin.encoders.uncertaingen import UncertainGenEncoder
from megobin.encoders.semibin_2 import SemiBin2Encoder
from megobin.evaluators.base import Evaluator
from megobin.evaluators.checkm2 import CheckM2Evaluator
from megobin.filters.base import Filter
from megobin.filters.no_op import NoOpFilter
from megobin.filters.uncertainty import UncertaintyFilter
from megobin.losses.base import ContrastiveLoss
from megobin.losses.hinge_contrastive import HingeContrastiveLoss
from megobin.losses.mahalanobis_bce import MahalanobisBCELoss
from megobin.trainers.base import Trainer
from megobin.trainers.single_phase import SinglePhaseTrainer
from megobin.trainers.two_phase import TwoPhaseTrainer
from megobin.utils.logger import Logger
from megobin.utils.no_op_logger import NoOpLogger
from megobin.utils.tensorboard_logger import TensorBoardLogger

def _encoder_inst() -> Encoder:
    return UncertainGenEncoder(input_dim=1, hidden_dim=1, embedding_dim=1)
def _encoder_semibin_inst() -> Encoder:
    return SemiBin2Encoder(input_dim=1, kmer_dim=1, hidden_dim=1, output_dim=1)
_loss_hinge: type[ContrastiveLoss] = HingeContrastiveLoss
_loss_mahalanobis: type[ContrastiveLoss] = MahalanobisBCELoss
_binner: type[Binner] = DBSCANEnsembleBinner
_evaluator: type[Evaluator] = CheckM2Evaluator
_sampler: type[PairSampler] = UncertainGenPairSampler
_sampler_semibin: type[PairSampler] = SemiBin2PairSampler
_filter_noop: type[Filter] = NoOpFilter
_filter_uncertainty: type[Filter] = UncertaintyFilter
_trainer_single: type[Trainer] = SinglePhaseTrainer
_trainer_two: type[Trainer] = TwoPhaseTrainer
_logger_noop: type[Logger] = NoOpLogger
_logger_tensorboard: type[Logger] = TensorBoardLogger
