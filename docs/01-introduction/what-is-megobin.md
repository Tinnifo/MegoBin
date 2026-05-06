# What MegoBin is

A modular pipeline for metagenomic binning. Every slot is swappable via YAML.

```
Dataset → Features (shared) → Encoder → Trainer → Binner → Evaluator
                                 ↑         ↑
                          Loss, Sampler   Optimizer, Scheduler, Logger
```

- **Encoder** — features → embeddings
- **Binner** — embeddings → cluster labels
- **Evaluator** — labels → completeness/contamination (CheckM2)

Run an experiment:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen
```

Swap a component:

```bash
python megobin/pipeline.py --config-name experiment/hybrid_uncertain_gen \
  encoder=semibin_encoder loss=hinge binner=dbscan_ensemble
```

Adding a new encoder, loss, or binner = **one Python file, one YAML, one test**.
