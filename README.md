# TGRL

PyTorch implementation of **Timetable Graph Reinforcement Learning for Distributed Flexible Job-Shop Scheduling with Dual-mode Transportation**.

TGRL combines two components:

- **Timetable-Lifted Reachability Graph (TRG):** timetable-conditioned successor relations, shared departure-event factors, and relaxed delivery bounds encode the downstream consequences of candidate assignments.
- **Graph-Counterfactual Optimization (GCO):** PPO uses frozen graph-based action-value control variates and compensation over all feasible actions, incorporating alternative assignments without additional counterfactual rollouts.

## Scheduling Model

DFJSP-DMT assigns each operation to an eligible machine, whose location determines the factory. Intra-factory transfers between operations depart immediately at dispatch; inter-factory transfers use a common periodic timetable. Dispatch exactly at a departure time catches that departure. Initial transfers from the source and final transfers to the depot depart immediately, independently of the timetable.

The selected machine is reserved from dispatch through departure waiting, travel, and processing. It is released at processing completion; final delivery does not extend its reservation. Dispatch requires a ready operation and an unreserved eligible machine. Time advances only when no feasible dispatch remains. Transportation capacity is nonbinding. The objective is **delivery makespan**, including every job's final transfer to the depot.

## Model Inputs

`transport_graph.py` constructs `tgrl_op` (8D), `tgrl_ma` (6D), `tgrl_pair` (8D), and successor-edge features (8D). These are distinct from the environment's raw features: keep `ope_feat_dim`, `ma_feat_dim`, and `pair_feat_dim` at **6, 4, and 4**.

The eight pair channels are processing time, travel duration, departure waiting, earliest-dispatch delay, inter-factory indicator, relaxed remaining delivery duration, delivery-bound increment, and mean successor departure waiting. Temporal pair channels use the fixed instance scale `max(1, p_max + travel_max + H)`; the inter-factory indicator is binary. Compatible but unavailable pairs provide graph context and are masked from action selection.

## Installation

Use Python 3.10 or newer. Install a PyTorch build appropriate for your CPU or CUDA device, then install the remaining requirements. The following installs the CPU build used for the release smoke checks:

```bash
python -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run commands from the repository root. Select `auto`, `cpu`, or a CUDA device such as `cuda:0` through `train_paras.device` and `test_paras.device`. Matplotlib and Visdom are optional and needed only for plotting or live visualization.

## Data

| Split | Directories | Instances per directory |
| --- | --- | --- |
| Validation | `100502`, `200502`, `151003`, `201003` under `data_dev/` | 100 |
| Synthetic test | `100502`, `200502`, `151003`, `201003`, `301003`, `401003` under `data_test/` | 100 |
| Public test | `data_test/Public_02/` | 130: Brandimarte (10), Rdata/Edata/Vdata (40 each) |

For example, `100502` denotes 10 jobs, 5 total machines, and 2 factories; `151003` denotes 15 jobs, 10 machines, and 3 factories. All included public instances use two factories. Machines are distributed approximately evenly across factories.

Synthetic and Brandimarte instances use integer-uniform intra-factory travel in `[0,10]`, inter-factory travel in `[10,20]`, and period `H` in `[10,20]`. Hurink instances use `[0,50]`, `[50,99]`, and `[50,99]`, respectively. Travel matrices are symmetric with zero diagonals; initial and final travel are sampled independently from the intra-factory range. These durations exclude departure waiting.

Training generates fresh instances through `env/case_generator.py`; validation and testing use the supplied files. `data_create.py` is an optional offline generator: edit its dimensions and output path before use. `case.py` converts original FJSP files, which are not included, into DFJSP-DMT instances. Its `auto` profile selects Brandimarte ranges for `Mk`-prefixed filenames and Hurink ranges otherwise:

```bash
python case.py --input-dir original_fjsp --output-dir generated_public --num-factories 2 --profile auto
```

Use `--profile brandimarte` or `--profile hurink` for renamed benchmarks. Generation replaces files with matching output names; keep the supplied evaluation sets unchanged for comparisons.

## Training

```bash
python train.py --config config.json
```

The supplied configuration trains on **10 jobs, 5 machines, and 2 factories** for 1,000 iterations, with 50 parallel instances, resampling every 50 iterations, and greedy validation every 10 iterations. It uses Adam with learning rate `3e-4`, 3 PPO epochs, minibatches of 512 transitions, discount 1, GAE parameter 0.98, and PPO clipping 0.2. GCO's control coefficient warms up to 0.3 over 20 updates. See `config_annotation.json` for field definitions.

The encoder uses width 32, 4 attention heads, 2 relational layers, and 2 assignment-factor layers. Each prediction head has two hidden layers of width 64. Training saves the configuration, logs, validation spreadsheets, and the best validated checkpoint under `save/train_.../`. Change `num_jobs`, `num_mas`, and `num_fac` together to select another training scale with a matching validation directory.

## Evaluation and Baselines

```bash
python test.py --config config.json
python PDRs.py --data-dirs data_test/100502 --output-dir PDRs
python OR-Tools.py --data-dirs data_test/100502 --time-limit 1800 --workers 8 --output-dir optimal
```

The default evaluation uses **all 130 instances in `Public_02` with best-of-100 sampling**. Set `test_paras.sample=false` for greedy decoding; `num_sample` controls the sampling budget and `num_average` controls repeated evaluations. Test dimensions are read from each file. `num_ins` limits the sorted file list, so use 100 for a complete synthetic test folder and 130 for the public folder.

The bundled checkpoint is `model/save_best_10_5_2_890.pt`. `rules=["TGRL"]` evaluates every `.pt` file in `model_dir`; alternatively list exact filenames. Set `model_dir` to a training output directory to evaluate a new model. Checkpoints must match `model_paras` and the 8D pair input; older 20D-pair checkpoints are incompatible.

Evaluation writes per-instance `makespan.xlsx` and `time.xlsx` under `save/test_.../`. Times cover graph construction, policy inference, and environment transitions, excluding initialization and feasibility checks. Sampling time covers the complete sampling budget. Public benchmark groups are identified by filename prefixes `Mk`, `r_`, `e_`, and `v_`; report their means separately. The paper's gap is `(mean_method - mean_CP-SAT) / mean_CP-SAT * 100`, rather than the mean of instance-wise gaps.

PDRs include FIFO-T, SPT-T, MOR-T, and MWKR-T. SPT-T minimizes the current reservation duration, while MOR-T prioritizes remaining-operation counts. CP-SAT uses the same reservation and timetable rules with non-delay dispatching and a default solver limit of **1800 seconds per instance**. Both baselines validate schedules and export CSV results. CP-SAT's `relative_gap` output is its solver optimality gap, not the paper's comparison gap. The optional `--allow-idle` mode solves a broader scheduling problem and saves results separately. Only `OPTIMAL` certifies optimality; fallback results are explicitly labeled.

## Repository Guide

- `DRL.py`, `transport_graph.py`, `embedding.py`: policy, GCO optimization, TRG construction, and graph encoding.
- `env/`: synthetic generation, instance parsing, and the batched scheduling environment.
- `train.py`, `validate.py`, `test.py`: training, validation, and evaluation entry points.
- `dfjsp_common.py`, `PDRs.py`, `OR-Tools.py`: shared baseline utilities and solvers.
- `config.json`, `config_annotation.json`: active settings and their descriptions.
- `case.py`, `data_create.py`: optional dataset preparation; `mlp.py` and `utils/` retain supporting components.

## References

This implementation builds on [fjsp-drl](https://github.com/songwenas12/fjsp-drl), [L2D](https://github.com/zcaicaros/L2D), and [MatNet](https://github.com/yd-kwon/MatNet). See `LICENSE` for the Apache License 2.0 terms.
