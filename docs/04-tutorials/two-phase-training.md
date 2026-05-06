# Tutorial: Two-phase training

UncertainGen trains the mean head first, then freezes it and trains the covariance head with a fresh optimizer.

## Why phased

End-to-end training lets the covariance inflate to absorb loss the mean should explain. Solution:

- **Phase 1** — train mean only, `include_std=False`, 50 epochs.
- **Phase 2** — freeze mean, train cov with a fresh Adam, `include_std=True`, 25 epochs.

## Config

`configs/trainer/two_phase.yaml`:

```yaml
_target_: megobin.trainers.two_phase.TwoPhaseTrainer
device: null
log_every: 50
logger: null
checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt
checkpoint_per_phase: false

phases:
  - params: mean
    epochs: 50
    batch_size: 10000
    grad_clip: null
    optimizer:
      _target_: torch.optim.Adam
      _partial_: true
      lr: 1e-3
    scheduler: null
    encoder_attrs: { include_std: false }
    loss_attrs:    { include_std: false }

  - params: cov
    epochs: 25
    batch_size: 10000
    grad_clip: null
    optimizer:
      _target_: torch.optim.Adam
      _partial_: true
      lr: 1e-3
    scheduler: null
    encoder_attrs: { include_std: true }
    loss_attrs:    { include_std: true }
```

Key points:
- `phases` is a list. Add a third phase = add a third entry.
- `params: mean | cov` keys into `encoder.parameter_groups()`.
- `_partial_: true` on optimizer = the trainer builds a **fresh** Adam per phase.
- `encoder_attrs` / `loss_attrs` flip flags via `setattr` at phase boundaries.

## What `fit` does

```python
for phase in self.phases:
    for k, v in phase.encoder_attrs.items(): setattr(encoder, k, v)
    for k, v in phase.loss_attrs.items(): setattr(loss_fn, k, v)

    active = encoder.parameter_groups()[phase.params]
    for p in encoder.parameter_groups()["all"]:
        p.requires_grad = (p in active)

    optimizer = phase.optimizer(active)               # fresh per phase
    scheduler = phase.scheduler(optimizer) if phase.scheduler else None

    for epoch in range(phase.epochs):
        for batch in DataLoader(sampler, ...):
            loss = encoder.training_step(batch, loss_fn)
            optimizer.zero_grad(); loss.backward()
            if phase.grad_clip:
                nn.utils.clip_grad_norm_(active, phase.grad_clip)
            optimizer.step()
        if scheduler: scheduler.step()
```

`checkpoint_per_phase: true` saves `encoder_phase_1.pt`, etc.

## Custom phase schedule

No new trainer class needed — override `phases` in your experiment YAML. Example three-phase schedule (pre-train, mean, cov):

```yaml
trainer:
  phases:
    - params: all
      epochs: 20
      batch_size: 10000
      optimizer: { _target_: torch.optim.Adam, _partial_: true, lr: 1e-3 }
      encoder_attrs: { include_std: false }
      loss_attrs:    { include_std: false }

    - params: mean
      epochs: 30
      batch_size: 10000
      optimizer: { _target_: torch.optim.Adam, _partial_: true, lr: 5e-4 }
      encoder_attrs: { include_std: false }
      loss_attrs:    { include_std: false }

    - params: cov
      epochs: 25
      batch_size: 10000
      optimizer: { _target_: torch.optim.Adam, _partial_: true, lr: 1e-3 }
      encoder_attrs: { include_std: true }
      loss_attrs:    { include_std: true }
```

Per-phase pair samplers aren't supported by `TwoPhaseTrainer` — write a new trainer if you need them.

## TensorBoard

Loss logs under `train/loss`, `train/epoch_loss`, plus phase-scoped `phase1/loss`, `phase2/loss`.

- Phase 1: smooth descent.
- Phase 2: a sharp jump at the boundary (fresh optimizer + cov head turning on), then a slower descent.

If Phase 2 doesn't drop, the Phase-1 mean was too good. Reduce `phase1.epochs`.

## Single-phase

`configs/trainer/single_phase.yaml`:

```yaml
defaults:
  - /optimizer@optimizer: adam
  - /scheduler@scheduler: step_lr
  - _self_

_target_: megobin.trainers.single_phase.SinglePhaseTrainer
epochs: 10
batch_size: 2048
grad_clip: null
params: all
checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt
checkpoint_every: null
```

`@optimizer` / `@scheduler` are Hydra package overrides — put `adam.yaml` under the `optimizer:` key.
