# MegoBin

MegoBin is a modular research pipeline for **metagenomic binning.** Every part of the pipeline — encoders, losses, pair samplers, trainers, binners, evaluators, loggers — is a swappable slot defined by a Python `Protocol` and composed via [Hydra](https://hydra.cc/) YAML configs, so adding a new method means writing one file and dropping in a config.

```
Dataset → Features (shared) → Encoder → Trainer → Binner → Evaluator
                                 ↑         ↑
                          Loss, Sampler   Optimizer, Scheduler, Logger
```

## Quickstart

```bash
mamba env create -f environment.yml
mamba activate megobin
pip install -e .

python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan
```

Swap a slot from the CLI:

```bash
python megobin/pipeline.py --config-name experiment/uncertain_gen_dbscan \
  binner=dbscan_ensemble loss=hinge trainer=single_phase
```

Output lands in `outputs/<date>/<time>/` — fully-resolved config, checkpoint, bin FASTAs, TensorBoard events. See [docs/02-getting-started/reading-the-output.md](docs/02-getting-started/reading-the-output.md).

## Slots

Each slot is a `@runtime_checkable` Protocol; configs select an implementation by name.

| Slot           | Source                                                | Configs                              |
|----------------|-------------------------------------------------------|--------------------------------------|
| Dataset        | [megobin/data/](megobin/data/)                        | [configs/dataset/](configs/dataset/) |
| Features       | [megobin/features/](megobin/features/)                | [configs/features/](configs/features/) |
| Encoder        | [megobin/encoders/](megobin/encoders/)                | [configs/encoder/](configs/encoder/) |
| Loss           | [megobin/losses/](megobin/losses/)                    | [configs/loss/](configs/loss/)       |
| Pair sampler   | [megobin/embedders/](megobin/embedders/)              | [configs/pair_sampler/](configs/pair_sampler/) |
| Trainer        | [megobin/trainers/](megobin/trainers/)                | [configs/trainer/](configs/trainer/) |
| Binner         | [megobin/binners/](megobin/binners/)                  | [configs/binner/](configs/binner/)   |
| Filter         | [megobin/filters/](megobin/filters/)                  | [configs/filter/](configs/filter/)   |
| Evaluator      | [megobin/evaluators/](megobin/evaluators/)            | [configs/evaluator/](configs/evaluator/) |

Adding a new component = one Python file + one YAML + one test. See [docs/04-tutorials/add-a-new-encoder.md](docs/04-tutorials/add-a-new-encoder.md).

## Documentation

- **Introduction** — [what is MegoBin](docs/01-introduction/what-is-megobin.md), [design philosophy](docs/01-introduction/design-philosophy.md)
- **Getting started** — [installation](docs/02-getting-started/installation.md), [reading the output](docs/02-getting-started/reading-the-output.md)
- **Architecture** — [slots and protocols](docs/03-architecture/slots-and-protocols.md), [Hydra configs](docs/03-architecture/hydra-configs.md)
- **Experiment tracking** — [TensorBoard](docs/05-experiment-tracking/tensorboard.md), [checkpoints + DVC](docs/05-experiment-tracking/checkpoints-and-dvc.md)
- **Testing** — [testing new components](docs/07-testing/testing-new-components.md)

## Requirements

Python ≥3.10. Core deps (PyTorch, Hydra, scikit-learn, etc.) install via `environment.yml`. The marker-aware DBSCAN binner optionally uses `hmmsearch`; CheckM2 is optional for evaluation. Full details in [installation.md](docs/02-getting-started/installation.md).
