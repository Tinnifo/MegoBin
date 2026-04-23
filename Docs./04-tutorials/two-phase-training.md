# Tutorial: Two-phase training

UncertainGen's training regime is not a single-optimizer loop. The encoder has two heads — a mean head and a covariance head — and the paper trains them sequentially: mean first for 50 epochs, then the mean is frozen and the covariance is trained for 25 more with a **fresh** Adam optimizer. This tutorial walks through how MegoBin supports that, and how you would write your own N-phase schedule.

If you've only seen `SinglePhaseTrainer`, this chapter is where the `parameter_groups` method on the Representation Protocol finally pays off.

## Why phased training

A diagonal Gaussian encoder can be trained end-to-end with a single Mahalanobis BCE loss — but in practice the mean head and covariance head fight each other. The mean wants to push pairs together/apart by moving means; the covariance wants to explain residual uncertainty. If both update simultaneously, the covariance finds a low-energy solution where it inflates to absorb loss the mean head should have been responsible for.

The fix is to train in two phases:

**Phase 1 — mean only.** Freeze the covariance head, train the mean head for 50 epochs with `include_std=False`. The encoder learns embeddings, ignoring uncertainty.

**Phase 2 — covariance only.** Freeze the mean head, train the covariance head for 25 epochs with `include_std=True` and a **fresh** Adam optimizer. The covariance learns to explain residual pair uncertainty on top of a fixed mean.

This generalizes — nothing in the trainer hardcodes "two phases" or "mean and cov". It is a list of phases, each with its own `params` name, epochs, optimizer, scheduler, and attribute overrides.

## The config that expresses this

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
    num_workers: 0
    shuffle: true
    optimizer:
      _target_: torch.optim.Adam
      _partial_: true
      lr: 1e-3
    scheduler: null
    encoder_attrs:
      include_std: false
    loss_attrs:
      include_std: false

  - params: cov
    epochs: 25
    batch_size: 10000
    grad_clip: null
    num_workers: 0
    shuffle: true
    optimizer:
      _target_: torch.optim.Adam
      _partial_: true
      lr: 1e-3
    scheduler: null
    encoder_attrs:
      include_std: true
    loss_attrs:
      include_std: true
```

The structure is worth dissecting.

`phases` is a list. Each entry is one phase. There's no "two-ness" hardcoded — swap to a three-phase schedule by adding a third entry and nothing else changes.

`params: mean` and `params: cov` are keys into the encoder's `parameter_groups()` dict. `UncertainGenRepresentation.parameter_groups()` returns `{"mean": [...], "cov": [...], "all": [...]}`, and the trainer uses the phase's `params` to pick which group is trainable for that phase; everything else is frozen.

`optimizer` uses `_partial_: true` — so the trainer receives a factory, not a bound optimizer. When phase 2 starts, the trainer calls `factory(encoder.parameter_groups()["cov"])` to build a **fresh** Adam bound to the covariance parameters only. This matters because optimizer state (Adam's first and second moments) should not carry over between phases — carrying them would leak Phase-1 dynamics into Phase-2.

`encoder_attrs` and `loss_attrs` let you flip flags on the encoder and loss at phase boundaries. `include_std: false` in Phase 1 tells UncertainGen to drop the covariance head from its forward and tells MahalanobisBCELoss to run in plain-embedding mode; `include_std: true` in Phase 2 flips both back on.

## What the trainer actually does

Pseudocode for `TwoPhaseTrainer.fit`:

```python
def fit(self, encoder, sampler, loss_fn):
    for phase in self.phases:
        # 1. Apply attribute overrides
        for k, v in phase.encoder_attrs.items():
            setattr(encoder, k, v)
        for k, v in phase.loss_attrs.items():
            setattr(loss_fn, k, v)

        # 2. Freeze everything not in the active parameter group
        active_group = encoder.parameter_groups()[phase.params]
        all_params = encoder.parameter_groups()["all"]
        for p in all_params:
            p.requires_grad = (p in active_group)

        # 3. Instantiate a FRESH optimizer bound to the active group
        optimizer = phase.optimizer(active_group)
        scheduler = phase.scheduler(optimizer) if phase.scheduler else None

        # 4. Standard training loop for this phase
        loader = DataLoader(sampler, batch_size=phase.batch_size, ...)
        for epoch in range(phase.epochs):
            for batch in loader:
                loss = encoder.training_step(batch, loss_fn)
                optimizer.zero_grad()
                loss.backward()
                if phase.grad_clip:
                    nn.utils.clip_grad_norm_(active_group, phase.grad_clip)
                optimizer.step()
            if scheduler:
                scheduler.step()
            # log per-epoch
        # save checkpoint per phase if configured

    # save final checkpoint
```

Four important mechanics:

First, only parameters in the active group have `requires_grad=True` during a phase. Everything else is frozen. If the encoder has BN layers in the frozen section, you'll also want to put the encoder in `eval()` mode for frozen sub-modules — the shipped `TwoPhaseTrainer` handles this.

Second, the optimizer is built per phase, not once at the top. That's the entire point of `_partial_: true` in the optimizer config.

Third, `encoder_attrs` and `loss_attrs` are free-form — the trainer uses `setattr`, which means any attribute the encoder or loss exposes can be flipped. `include_std` is the specific case we use, but it generalizes: imagine a three-phase schedule where Phase 3 turns on a different loss weighting term.

Fourth, `checkpoint_per_phase: true` saves an intermediate checkpoint after each non-final phase. Useful when you want to diagnose "did the mean converge before we moved on?" — load the phase-1 checkpoint, inspect, decide.

## Writing a custom phase schedule

You don't need a new trainer class for a new schedule — you just need a new experiment config that overrides the `phases` list. Example: a 3-phase schedule where you pre-train on random pairs, fine-tune on semantically-matched pairs, and finally fit the covariance head.

In your experiment YAML:

```yaml
trainer:
  phases:
    - params: all
      epochs: 20
      batch_size: 10000
      optimizer:
        _target_: torch.optim.Adam
        _partial_: true
        lr: 1e-3
      encoder_attrs: { include_std: false }
      loss_attrs:    { include_std: false }

    - params: mean
      epochs: 30
      batch_size: 10000
      optimizer:
        _target_: torch.optim.Adam
        _partial_: true
        lr: 5e-4
      encoder_attrs: { include_std: false }
      loss_attrs:    { include_std: false }

    - params: cov
      epochs: 25
      batch_size: 10000
      optimizer:
        _target_: torch.optim.Adam
        _partial_: true
        lr: 1e-3
      encoder_attrs: { include_std: true }
      loss_attrs:    { include_std: true }
```

Note you would **also** need two different pair samplers — one for the pre-training phase and one for the fine-tuning phase. That's not something the current `TwoPhaseTrainer` supports out of the box; you'd either need to concatenate the samplers or write a new trainer. The Protocol makes the latter cheap.

## How this looks in TensorBoard

Loss curves log under `train/loss` and `train/epoch_loss` for all phases, plus `phase1/loss`, `phase2/loss` as phase-scoped variants so you can see the two regimes separately. Visually, you'll see:

- Phase 1: smooth descent over 50 epochs as the mean head converges.
- Phase 2: **a discontinuity** at the phase boundary — a fresh optimizer plus the covariance head coming online usually shows as a sharp jump in loss before a second, slower descent. That jump is expected.

If your phase 2 loss **doesn't** drop meaningfully, your covariance head isn't learning — usually a sign the Phase-1 mean solution was too good to leave residual variance to explain. Dropping `phase1.epochs` to something smaller is the usual fix.

## Single-phase case

For completeness, `configs/trainer/single_phase.yaml`:

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
num_workers: 0
shuffle: true
device: null
log_every: 50
logger: null

checkpoint_path: ${hydra:runtime.output_dir}/encoder.pt
checkpoint_every: null
```

This is what SemiBin uses. One phase, `params: all`, one optimizer, one scheduler. The `@optimizer` syntax in the defaults is a Hydra "package override" — it tells Hydra to put the `adam.yaml` contents into the `optimizer` key (rather than at the top level, which would be the default behaviour of that file). Same for `scheduler`.

## Summary

Two-phase training in MegoBin is a config pattern, not a hardcoded concept. The `TwoPhaseTrainer` accepts a list of phases; each phase names a parameter group, builds a fresh optimizer via `_partial_`, and optionally flips attributes on the encoder and loss. Adding a third phase is five lines of YAML. Adding a completely new training strategy is a new trainer class that satisfies the one-method `Trainer` Protocol — no pipeline changes.
