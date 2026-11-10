# What MegoBin is

A modular pipeline for metagenomic binning. Every slot is swappable via YAML.

```
Dataset → features → embedder → filter → binner → [refiner] → bins/
                       ↑
          Sampler + Encoder + Loss + Trainer (Optimizer, Scheduler, Logger)

bins/ → python -m megobin.evaluate → completeness / contamination
```

- **Encoder** — features → embeddings
- **Binner** — embeddings → cluster labels
- **Evaluator** — a separate entry point (`python -m megobin.evaluate`): bins directory → completeness/contamination ([CheckM2](https://github.com/chklovski/CheckM2)). It is not part of the method run, so every method is judged the same way.

A run is a DAG of stages with declared `needs` / `provides`; the order is resolved from the target and all contracts are checked before any compute. See [Contracts and stages](../03-architecture/contracts-and-stages.md).

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
