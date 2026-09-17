# STGCNN_baseline

A reproduction of:

> Zhang, Y., Zhong, R., & Mahadevan, S. (2022). "Airport surface movement prediction and
> safety assessment with spatial-temporal graph convolutional neural network."
> *Transportation Research Part C: Emerging Technologies*, 144, 103873.

used as a comparison-method baseline in the thesis's trajectory-prediction results table.
It is trained/evaluated on the exact same scenes, splits, agent filtering, and coordinate
normalization as this repo's other `AmeliaTF_main`-family methods -- only the model
(STG-CNN + TXP-CNN) and its training loop are new.

This is a **unimodal** baseline: a single bivariate Gaussian per agent per future
timestep, no turn-mode classification. Only an aggregate "All" ADE/FDE/NLL is reported
(no per-mode Hold/Straight/TurnLeft/TurnRight breakdown).

## What was copied vs. what is new

Copied **unmodified** from `AmeliaTF_main` (so data loading is byte-for-byte identical):
- `amelia_scenes/`
- `amelia_tf/data/` (dataset + datamodule)
- `amelia_tf/utils/` (not explicitly requested, but required by the above: e.g.
  `data_utils.py`, `global_masks.py`, `metrics.py`, `pylogger.py`, `utils.py`, ...).
  This follows the same "each top-level folder is a fully self-contained copy" pattern
  every other sibling folder (`AmeliaTF_main_two_phases`, `AmeliaTF_main_two_phases_4T`,
  ...) already uses, and is necessary for the dataset/datamodule classes to import.
- `configs/data/{default,kbos,klax,kmsy}.yaml`, `configs/paths/default.yaml`,
  `configs/callbacks/*`, `configs/trainer/*`, `configs/logger/wandb.yaml`,
  `configs/extras/default.yaml`, `configs/hydra/default.yaml` -- needed for Hydra
  composition, again unmodified.

New:
- `amelia_tf/models/components/stgcnn.py` -- graph construction (`build_adjacency`) +
  the STG-CNN spatio-temporal graph-conv layer.
- `amelia_tf/models/components/txpcnn.py` -- the TXP-CNN time-extrapolator.
- `amelia_tf/models/stgcnn_traj_pred.py` -- `STGCNNPredictor` (ties the two components
  together + a final head projecting to the 4 diagonal-Gaussian parameters) and
  `STGCNNTrajPred` (the `LightningModule`: `model_step`/`training_step`/
  `validation_step`/`test_step`, NLL loss, ADE/FDE logging at horizons 20 and 50).
- `configs/model/stgcnn.yaml`, `configs/train_stgcnn_{kbos,klax,kmsy}.yaml`
- `amelia_tf/train_stgcnn_{kbos,klax,kmsy}.py`, `train_stgcnn_{kbos,klax,kmsy}.sh`
- `configs/data/default2.yaml`, `configs/data/{kbos,klax,kmsy}2.yaml`, `configs/paths/default2.yaml`,
  `configs/train_stgcnn_{kbos,klax,kmsy}_20.yaml`, `amelia_tf/train_stgcnn_{kbos,klax,kmsy}_20.py`,
  `train_stgcnn_{kbos,klax,kmsy}_20.sh` -- a genuinely separate 20s-horizon training run
  (`traj_len: 30`, `pred_lens: [10, 20]`, its own `proc_full_scenes2/` scenes dir and
  `splits2/` split cache), following the same no-suffix-is-50s / `2`-suffix-is-20s
  convention as `AmeliaTF_main_two_phases_4T`. This is distinct from the `t=20` metric
  already logged by the 50s-config run above: that is an intermediate-horizon readout
  from a model trained on 60-frame (10 hist + 50 pred) sequences, not a model trained
  end-to-end for a 20s prediction task.

## How to launch training (HPC / SLURM)

```bash
cd STGCNN_baseline
sbatch train_stgcnn_kbos.sh
sbatch train_stgcnn_klax.sh
sbatch train_stgcnn_kmsy.sh

# 20s-horizon counterparts (separate training run, see "What was copied vs. what is new"):
sbatch train_stgcnn_kbos_20.sh
sbatch train_stgcnn_klax_20.sh
sbatch train_stgcnn_kmsy_20.sh
```

Or directly, e.g. on an interactive GPU node:

```bash
python -m amelia_tf.train_stgcnn_kmsy
```

Each `.sh` mirrors the exact SBATCH header (`partition=sae`, `account=pilot_sae_gpu`,
`gres=gpu:1`, module-load lines, `conda activate amelia_env`) used by
`AmeliaTF_main/train_kmsy.sh` etc. -- only the python invocation line changed. Before
launching, run `wandb login` once on the cluster (see the credential note below).

To sweep `num_txp_layers` (see the "5 or 7" simplification below), override it on the
command line, e.g.:

```bash
python -m amelia_tf.train_stgcnn_kmsy model.net.num_txp_layers=7
```

## Judgment calls and simplifications (read before citing these numbers)

1. **NLL / Gaussian parameterization -- resolved: diagonal, no rho.** `AmeliaTF_main/
   amelia_tf/models/components/gmm.py` (the project's existing GMM head) parameterizes
   each mixture component with only `(mu_x, mu_y, sigma_x, sigma_y)` -- no correlation
   term -- and computes its regression loss via `torch.nn.functional.gaussian_nll_loss`
   (a diagonal, independent-axes Gaussian; see `amelia_tf/utils/losses.py::marginal_loss`,
   which calls `F.gaussian_nll_loss(mu, target, sigma**2)`). The paper's Eq. 1 technically
   specifies a 5th parameter `rho` (correlation) for a proper bivariate Gaussian, but per
   the user's decision this baseline **drops rho** and uses the same diagonal
   `gaussian_nll_loss` formula as the rest of the codebase (`diagonal_gaussian_nll` in
   `stgcnn_traj_pred.py`), so `val/nll`/`test/nll` here are directly comparable to every
   other method's NLL in the same table. The sigma activation convention
   (`F.softplus(raw_sigma) + 1e-3`) still matches `gmm.py` verbatim. The output head is
   4-dimensional (`mu_x, mu_y, sigma_x, sigma_y`), not 5.

2. **High-speed / runway exclusion -- resolved: no filter needed.** The paper (Sec. 3,
   data prep) excludes agents with speed > 20 knots from training. Per the user: the
   scene-generation pipeline that produces this project's dataset already excludes
   runway trajectories upstream (before any of these `AmeliaTF_main*` folders' code ever
   sees the data), so no additional in-pipeline filter is needed here -- this baseline
   trains on the same effective population as every other method in the table, and that
   already matches the paper's intent (taxiway-only ground movement). No code change was
   required for this point.

3. **STG-CNN input window / no zero-padded future.** `AmeliaTF_main`'s transformer
   (`trajpred.py::model_step`) feeds the network a full `(hist_len + max_pred_len)`-long
   sequence with the future zeroed out (`X[:, :, :hist_len] = Y[:, :, :hist_len]`), since
   its architecture is a masked, autoregressive-style model. STG-CNN + TXP-CNN is an
   *extrapolation* architecture (per the paper and Social-STGCNN): STG-CNN only ever sees
   the `hist_len` observed timesteps, and TXP-CNN maps that fixed-length embedding
   directly to `max_pred_len` future timesteps by reinterpreting the time axis as a
   convolutional channel axis. I therefore slice `X = rel_sequences[:, :, :hist_len]`
   (no zero-padding) rather than mimicking the transformer's masking pattern, and index
   ADE/FDE/NLL horizons directly as `[:, :, :t]` into the future-only tensors (rather
   than `AmeliaTF`'s `[:, :, :hist_len+t]` into a history+future tensor). This is a
   structural necessity, not a simplification, but flagging it since it means
   `model_step`'s tensor layout looks different from `trajpred.py`'s at first glance.

4. **`num_txp_layers` not re-tuned per airport.** The paper reports the best number of
   TXP-CNN layers differs by airport (5 or 7). Per your instruction, this reproduction
   exposes `num_txp_layers` as a config value (`configs/model/stgcnn.yaml`, default `5`)
   rather than re-running a per-airport sweep. Override with
   `model.net.num_txp_layers=7` to try the alternative.

5. **No map/context encoder.** The paper's STG-CNN + TXP-CNN has no semantic-map
   encoder at all (only agent-graph state). `add_context: false` is set in each
   `train_stgcnn_*.yaml` purely to skip the dataset's (otherwise unused) map-context/
   polyline-adjacency computation -- this does **not** change which scenes, splits, or
   agents are used, only whether an unused tensor gets computed. This is arguably *more*
   faithful to the paper than leaving context on, not a simplification.

6. **No visualization / off-road evaluation / modal-classification machinery.**
   `trajpred.py` in `AmeliaTF_main` also computes off-road-distance metrics, per-mode
   breakdowns, a balanced test set, and scene-plotting. None of that is required by the
   paper or by your stated scope ("unimodal... only report an aggregate 'All' row"), so
   `stgcnn_traj_pred.py` omits it entirely to keep the file short and readable. If the
   comparison table later wants off-road metrics for this baseline too, the mechanism to
   reuse is `amelia_tf/utils/off_road_evaluator.py` (copied over, unused).

7. **Optimizer hyperparameters beyond lr.** The paper specifies SGD, lr=0.02, batch
   128, 300 epochs (Sec. 4.1) but does not report momentum or weight decay. I set
   `momentum: 0.9, weight_decay: 0.0` in `configs/model/stgcnn.yaml` as an explicit,
   overridable default rather than leaving momentum at PyTorch's SGD default of 0
   (plain SGD with no momentum converges very slowly / can be unstable at lr=0.02 for
   this kind of model) -- **worth a second look / a quick sweep** if training is
   unstable at that learning rate.

8. **Adjacency for padded/invalid agents.** `build_adjacency` (in `stgcnn.py`) zeroes
   every edge touching a node the dataset's existing `agent_masks` marks invalid at that
   timestep (padded agent slots when a scene has fewer than `k_agents` real objects, or
   real agents with interpolated/missing data at a given timestep). The paper doesn't
   discuss missing-data handling for the graph explicitly; this is the natural way to
   integrate the paper's graph construction with this project's existing
   `agent_masks` convention, and prevents padding artifacts (all agents' padded slots
   sit at position (0,0)) from injecting bogus edges.

9. **W&B credential.** `AmeliaTF_main`'s `train_*.py` scripts call
   `wandb.login(key="...")` with a literal API key hard-coded in source. That key is
   intentionally **not** reproduced in the new `train_stgcnn_*.py` scripts (flagged as a
   credential-leakage risk when writing these files) -- they call `wandb.login()`
   instead, which picks up a cached login or the `WANDB_API_KEY` env var. Run
   `wandb login` once per HPC account before submitting the `.sh` jobs.

## Wandb key naming (for cross-method comparison)

Mirrors `trajpred.py`'s convention directly:
`losses/train`, `losses/val`, `losses/test`, `val/ade/t=20`, `val/ade/t=max`,
`val/fde/t=20`, `val/fde/t=max`, `val/nll`, and the `test/...` equivalents
(`test/ade/t=20`, `test/ade/t=max`, `test/fde/t=20`, `test/fde/t=max`, `test/nll`).
`t=max` is used for the longest horizon (50) instead of `t=50`, exactly matching
`trajpred.py` and the checkpoint/early-stopping monitor key (`val/ade/t=max`) already
configured in the copied `configs/callbacks/default.yaml`.

## Things to double-check before trusting these results

- `hidden_channels=64` for the STG-CNN embedding and the TXP-CNN kernel sizes (3x3) are
  reasonable defaults but are **not** specified by the paper -- worth a small sweep if
  the baseline underperforms suspiciously.
- SGD momentum/weight_decay (point 7) were not in the paper; sanity-check training
  stability at lr=0.02.
- I could not run any of this (no GPU/data in this environment) -- the very first HPC
  run should be watched for shape errors, since the STG-CNN/TXP-CNN plumbing was only
  checked analytically (see the docstrings in `stgcnn.py`/`txpcnn.py`/
  `stgcnn_traj_pred.py`), not executed.
