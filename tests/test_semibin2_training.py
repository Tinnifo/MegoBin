"""SemiBin2 training details: loss arithmetic and the trailing partial batch."""

import functools

import numpy as np
import torch

from megobin.data.semibin2_sampler import SemiBin2PairSampler
from megobin.encoders.semibin_2 import SemiBin2Encoder
from megobin.losses.hinge_contrastive import HingeContrastiveLoss
from megobin.trainers.single_phase import SinglePhaseTrainer


def _reference_loss(e1: torch.Tensor, e2: torch.Tensor, label: torch.Tensor):
    """SemiBin's loss_function, written out as SemiBin calls it (float64)."""
    e1, e2, label = e1.double(), e2.double(), label.double()
    d = torch.norm(e1 - e2, p=2, dim=1)
    return torch.mean(
        label * torch.square(d) + (1 - label) * torch.square(torch.relu(1 - d))
    )


def test_loss_equals_semibin_formula_in_float64():
    torch.manual_seed(0)
    z_i = (0.3 * torch.randn(64, 100)).requires_grad_()
    z_j = 0.3 * torch.randn(64, 100)
    label = (torch.rand(64) < 0.5).float()

    loss = HingeContrastiveLoss(margin=1.0)(z_i, z_j, label)
    assert loss.dtype == torch.float64
    ref_in = z_i.detach().clone().requires_grad_()
    ref = _reference_loss(ref_in, z_j, label)
    assert loss.item() == ref.item()

    loss.backward()
    ref.backward()
    assert z_i.grad is not None and ref_in.grad is not None
    assert torch.equal(z_i.grad, ref_in.grad)


def test_loss_is_zero_with_finite_gradient_for_identical_must_link_pair():
    z = torch.zeros(4, 100, requires_grad=True)
    loss = HingeContrastiveLoss()(z, z.detach().clone(), torch.ones(4))
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


class _CountingLoss(HingeContrastiveLoss):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, z_i, z_j, label):
        self.batch_sizes.append(int(label.shape[0]))
        return super().forward(z_i, z_j, label)


def _fit(drop_last: bool) -> list[int]:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    split = rng.random((6, 8), dtype=np.float32)  # 3 contigs, two halves each
    whole = rng.random((5, 8), dtype=np.float32)
    sampler = SemiBin2PairSampler(
        features_split=split, features_whole=whole, ratio=3, seed=0
    )
    assert len(sampler) == 12  # 3 must-link + 9 cannot-link
    encoder = SemiBin2Encoder(input_dim=8, kmer_dim=8, hidden_dim=16, output_dim=4)
    loss = _CountingLoss()
    trainer = SinglePhaseTrainer(
        optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
        epochs=2,
        batch_size=5,
        drop_last=drop_last,
        device="cpu",
    )
    trainer.fit(encoder, sampler, loss)
    return loss.batch_sizes


def test_trailing_partial_batch_is_dropped_when_asked():
    assert _fit(drop_last=True) == [5, 5, 5, 5]


def test_trailing_partial_batch_is_kept_by_default():
    assert _fit(drop_last=False) == [5, 5, 2, 5, 5, 2]
