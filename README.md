# MegoBin

MegoBin is a modular research pipeline for **metagenomic binning.** Every part of the pipeline — encoders, losses, pair samplers, trainers, binners, evaluators, loggers — is a swappable slot defined by a Python `Protocol` and composed via [Hydra](https://hydra.cc/) YAML configs, so adding a new method means writing one file and dropping in a config.

```
Dataset → Features (shared) → Encoder → Trainer → Binner → Evaluator
                                 ↑         ↑
                          Loss, Sampler   Optimizer, Scheduler, Logger
```
