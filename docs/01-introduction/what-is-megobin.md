# What MegoBin actually is

A Python package plus configs plus tests, organized as three swappable slots wired together by a single Hydra-driven entry point.

```
Dataset → Features (shared) → Encoder → Trainer → Binner → Evaluator
                                 ↑         ↑
                          Loss, Sampler   Optimizer, Scheduler, Logger
```

The **Encoder** turns features into embeddings. The **Binner** turns embeddings into cluster labels. The **Evaluator** turns cluster labels into completeness/contamination scores via CheckM2. Everything that feeds those three slots — features, losses, pair samplers, trainers, optimizers, schedulers, loggers — is itself a swappable component backed by a Python `Protocol` and composed via YAML.

A single command runs a full experiment:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen
```

And a single override swaps any component without editing code:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen \
  encoder=semibin_encoder loss=hinge binner=dbscan_ensemble
```

The goal is fast iteration. Adding a new encoder or loss or binner should mean **one file, one YAML, one test** — not a refactor of the pipeline. The rest of this guide is about how the architecture delivers on that goal, and how you work inside it.
