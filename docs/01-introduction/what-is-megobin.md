# What MegoBin is

A modular pipeline for metagenomic binning. Every slot is swappable via YAML.

```
Dataset → Features (shared) → Encoder → Trainer → Binner → Evaluator
                                 ↑         ↑
                          Loss, Sampler   Optimizer, Scheduler, Logger
```

- **Encoder** — features → embeddings
- **Binner** — embeddings → cluster labels
- **Evaluator** — labels → completeness/contamination ([CheckM2](https://github.com/chklovski/CheckM2))

Run an experiment:

```bash
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan
```

Swap a component:

```bash
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan \
  binner=dbscan_ensemble filter=uncertainty
```

Adding a new encoder, loss, or binner = **one Python file, one YAML, one test**.
