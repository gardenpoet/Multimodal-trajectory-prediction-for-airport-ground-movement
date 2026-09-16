# Two-Phases retraining grid (no weighting), 3 airports x 3 methods x 2 horizons

This documents the HPC-ready training grid added in this pass: **kbos / klax /
kmsy** x **1T / 2T / 4T** x **20s / 50s** = **18 trajectory runs**, each
preceded by a one-time mode-classifier training per airport+horizon (**6**
more jobs), for **24 SLURM submissions** total. All 24 jobs disable
mode-frequency loss weighting in both stages (the non "-W" variant).

Read the "Please double-check before submitting" section near the bottom
before launching anything — it flags one significant, evidence-based but
unverified fix (the 20s/50s data wiring) that changes which physical dataset
each run trains against.

## 1. The grid

| Airport | Horizon | num_futures | Mode-training config | Trajectory-sweep config | Submit mode job | Submit trajectory job |
|---|---|---|---|---|---|---|
| kbos | 50s | - | `configs/train_two_stage_kbos.yaml` | - | `sbatch train_mode_kbos_50.sh` | - |
| kbos | 50s | 1 | - | `configs/train_two_stage_kbos.yaml` | - | `sbatch --export=ALL,NUM_FUTURES=1 train_traj_kbos_50.sh` |
| kbos | 50s | 2 | - | `configs/train_two_stage_kbos.yaml` | - | `sbatch --export=ALL,NUM_FUTURES=2 train_traj_kbos_50.sh` |
| kbos | 50s | 4 | - | `configs/train_two_stage_kbos.yaml` | - | `sbatch --export=ALL,NUM_FUTURES=4 train_traj_kbos_50.sh` |
| kbos | 20s | - | `configs/train_two_stage_kbos_20.yaml` | - | `sbatch train_mode_kbos_20.sh` | - |
| kbos | 20s | 1 | - | `configs/train_two_stage_kbos_20.yaml` | - | `sbatch --export=ALL,NUM_FUTURES=1 train_traj_kbos_20.sh` |
| kbos | 20s | 2 | - | `configs/train_two_stage_kbos_20.yaml` | - | `sbatch --export=ALL,NUM_FUTURES=2 train_traj_kbos_20.sh` |
| kbos | 20s | 4 | - | `configs/train_two_stage_kbos_20.yaml` | - | `sbatch --export=ALL,NUM_FUTURES=4 train_traj_kbos_20.sh` |
| klax | 50s | - | `configs/train_two_stage_klax.yaml` | - | `sbatch train_mode_klax_50.sh` | - |
| klax | 50s | 1/2/4 | - | `configs/train_two_stage_klax.yaml` | - | `sbatch --export=ALL,NUM_FUTURES={1,2,4} train_traj_klax_50.sh` |
| klax | 20s | - | `configs/train_two_stage_klax_20.yaml` | - | `sbatch train_mode_klax_20.sh` | - |
| klax | 20s | 1/2/4 | - | `configs/train_two_stage_klax_20.yaml` | - | `sbatch --export=ALL,NUM_FUTURES={1,2,4} train_traj_klax_20.sh` |
| kmsy | 50s | - | `configs/train_two_stage_kmsy.yaml` | - | `sbatch train_mode_kmsy_50.sh` | - |
| kmsy | 50s | 1/2/4 | - | `configs/train_two_stage_kmsy.yaml` | - | `sbatch --export=ALL,NUM_FUTURES={1,2,4} train_traj_kmsy_50.sh` |
| kmsy | 20s | - | `configs/train_two_stage_kmsy_20.yaml` | - | `sbatch train_mode_kmsy_20.sh` | - |
| kmsy | 20s | 1/2/4 | - | `configs/train_two_stage_kmsy_20.yaml` | - | `sbatch --export=ALL,NUM_FUTURES={1,2,4} train_traj_kmsy_20.sh` |

Order matters: **run the mode job for an airport+horizon before submitting its
three trajectory-sweep jobs** (they load the mode checkpoint the mode job
produces). The three trajectory-sweep jobs for a given airport+horizon are
otherwise independent of each other and can run in parallel.

## 2. Checkpoint-reuse workflow (mode trained once per airport+horizon)

Mode classification does not depend on `num_futures` (1T/2T/4T runs of the
same airport+horizon produce identical mode-classification accuracy, verified
empirically earlier this project), so it is wasteful to retrain it 3x. This
was **not previously wired up in the airport-specific scripts** — it existed
only in the generic, seemingly-unused `amelia_tf/train_two_stage.py` /
`configs/train_two_stage.yaml` (flag `skip_mode_training`, method
`load_pretrained_mode_model`), and had been dropped when
`train_two_stage_{kbos,klax,kmsy}.py` were forked from it.

I ported and extended this mechanism into
`amelia_tf/train_two_stage_{kbos,klax,kmsy}.py` and the new
`_20.py` siblings:

- **`train_mode_model()`** (Stage 1), after saving the best checkpoint, now
  also copies it to the fixed `cfg.mode_ckpt_path` location (creating parent
  dirs as needed). This is new: previously the best checkpoint only lived
  under Hydra's per-run timestamped output directory, which the next run
  couldn't discover.
- **`load_pretrained_mode_model(checkpoint_path)`** (new method): validates
  the fixed checkpoint exists and points `self.mode_ckpt_path` at it, skipping
  Stage 1 entirely.
- **`run()`**: now checks `cfg.skip_mode_training` (new flag, default
  `False`) to choose between `train_mode_model()` and
  `load_pretrained_mode_model()`, and a new `cfg.skip_traj_training` flag
  (default `False`) that, if `True`, returns immediately after Stage 1 -
  this is what makes the dedicated "mode-only" SLURM jobs cheap.

  I deliberately used `False` as the default for both flags (train-two-stage.py's
  own `skip_mode_training` defaulted to `True`, which struck me as a
  dangerous default to inherit — a config that forgets to set it would
  silently skip real training). Every config in this grid sets these flags
  explicitly for its intended role.

Workflow per airport+horizon:
1. `sbatch train_mode_{airport}_{horizon}.sh` - trains Stage 1, copies the
   checkpoint to e.g. `${ckpt_dir}/Single-Airport/kbos2/mode_model/kbos2_twophases_50.ckpt`,
   exits.
2. Once (1) finishes, submit the three `train_traj_{airport}_{horizon}.sh`
   jobs with `NUM_FUTURES=1`, `2`, `4`. Each loads that fixed checkpoint via
   `skip_mode_training=true` and trains only Stage 2 + the Stage 3
   end-to-end test.

**Not verified by running code** (no GPU/training environment in this
session): the checkpoint copy/load round-trip (`shutil.copy2`, then
`_load_model_state`'s prefix-stripping load in Stage 3) has not been
exercised end-to-end. The logic mirrors the pre-existing, evidently-used
`_load_model_state` method exactly, so risk should be low, but please watch
the first mode-only job's log for the "Copied best mode checkpoint to fixed
reuse path" line before trusting the sweep jobs to find it.

## 3. Weighting disabled (both stages, one shared mechanism)

There is **no existing on/off config flag** for this - loss weighting was
unconditionally hardcoded on. Concretely:

- `amelia_tf/data/datamodule.py` (previously ~line 474, now ~481): `setup()`
  unconditionally called `self.mode_frequency(self.data_train)` whenever
  `task_name == "train"`. `mode_frequency()` (originally lines 340-422)
  computes `self.mode_weights`, a per-turn-mode class-balancing tensor
  (`alpha=0.75` inverse-frequency weighting).
- `amelia_tf/models/traj_pred_combined.py`:
  - `ModePredictionModel.setup()` (originally lines 80-88) injects
    `dm.mode_weights` and **raises `RuntimeError`** if the datamodule doesn't
    have the attribute at all - it was mandatory, not optional. Used at line
    143: `F.cross_entropy(..., weight=self.mode_weights, ...)` (Stage 1 CE
    loss).
  - `TrajectoryPredictionModel.setup()` (originally lines 430-437) injects the
    **same** `dm.mode_weights` tensor. Used at (originally) line 477:
    `sample_weights = self.mode_weights[ego_true]`, then applied to the
    per-sample WTA/Gaussian-NLL loss at lines 515-516 and 559-560.

So Stage 1 and Stage 2 weighting are **not two independent switches** - both
consume the one tensor the datamodule computes. I added a single gate:

- `amelia_tf/data/datamodule.py`: new
  `self.use_mode_weights = getattr(self.eparams.data_prep, "use_mode_weights", True)`
  (added next to the existing `use_balanced_test` / `test_size` getattrs).
  `setup()` now only calls `mode_frequency()` if this is `True`; otherwise it
  logs and sets `self.mode_weights = None`.
- `amelia_tf/models/traj_pred_combined.py`: guarded the one place that would
  crash on `None` -
  `sample_weights = self.mode_weights[ego_true] if self.mode_weights is not None else None`.
  (`F.cross_entropy(weight=None, ...)` in `ModePredictionModel` is already
  PyTorch's standard unweighted form, so no change was needed there.)
- Default is `True` (backward-compatible for any other consumer of
  `DataModule` I'm not touching, e.g. `eval_two_stage_kbos.yaml`,
  `e2e_finetune`/`scratch_joint` variants). Every one of the 6
  `train_two_stage_*.yaml` configs in this grid **explicitly** sets it to
  `False` via:
  ```yaml
  data:
    extra_params:
      data_prep:
        use_mode_weights: false
  ```

**Not verified by running code**: I could not execute training to confirm
`use_mode_weights=false` actually leaves loss values sane (e.g. that no other
code path assumes `dm.mode_weights` is a tensor). I grepped every
`mode_weights` occurrence in `traj_pred_combined.py` (12 hits) and accounted
for all of them; I did not exhaustively search the rest of the codebase
(metrics/plotting utilities) for a stray `.mode_weights` access.

## 4. The 20s/50s horizon fix (please read before submitting)

**This is the change I'm least certain about and most want you to sanity
check against any past run logs / W&B history you have.**

I found that `configs/train_two_stage_{kbos,klax,kmsy}.yaml` (the existing,
already-in-use 50s configs) composed `data: {kbos,klax,kmsy}2.yaml`, which
resolves to `configs/data/default2.yaml`: `traj_len: 30, hist_len: 10,
pred_lens: [10, 20]`. `amelia_scenes/utils/dataset.py:40`
(`time_at_frame = base_time + timedelta(seconds=frame_idx)`) confirms 1
frame = 1 second, so `traj_len - hist_len` = 20 future timesteps = a **20
second** horizon - despite their checkpoint paths being labeled
`_twophases_50`. Meanwhile `configs/data/default.yaml` (`traj_len: 60`) gives
a 50-timestep/**50 second** future window, and is composed by `data:
{kbos,klax,kmsy}.yaml` (no "2" suffix).

I initially cited `configs/eval_two_stage_kbos.yaml` (which pairs `data:
kbos.yaml` with `paths: default3.yaml`) as corroborating evidence and set
the fixed mainline configs to use `paths: default3.yaml`. **This was wrong
and has been corrected** (caught by the user during review): the project's
actual convention is that `data`/`paths` suffixes must match one-to-one -
no suffix = 50s, suffix `2` = 20s - and `paths/default3.yaml` is an
unrelated, differently-numbered scratch scene cache (its `scenes_dir` is
`proc_full_scenes7`, not `...3`; the filename and content numbering don't
even agree with each other) left over from some other experiment, not part
of the 20s/50s pairing at all. `eval_two_stage_kbos.yaml`'s use of
`default3.yaml` was coincidental/unrelated, not evidence of the intended
50s pairing.

Given all that, the **existing** `train_two_stage_{kbos,klax,kmsy}.yaml`
had a data-wiring bug (using the 20s-shaped data while claiming to be the
50s config), fixed as follows:

- `data: {kbos,klax,kmsy}2.yaml` -> `data: {kbos,klax,kmsy}.yaml`
- `paths: default2.yaml` -> `paths: default.yaml` (matching the no-suffix
  data config - NOT `default3.yaml`)
- kept `_twophases_50` ckpt-path suffixes (now actually correct)

The new `configs/train_two_stage_{kbos,klax,kmsy}_20.yaml` use exactly what
the old 50s configs used to use (`data: {airport}2.yaml`, `paths:
default2.yaml`), now correctly labeled `_twophases_20`.

**Why this matters if I'm wrong:** if any `_50`-labeled checkpoint already on
your HPC disk was in fact trained on the 20s-shaped data (i.e. my "bug" theory
is wrong and the old wiring was intentional for some reason I couldn't see
from the config files alone), then the grid as configured now will retrain
that airport at a genuinely different horizon than before, under the same
"_50" checkpoint name pattern - not silently wrong, since you're retraining
everything anyway, but worth knowing going in. I did not find any comment,
README, or commit message in this repo explaining the "2" suffix convention,
so this is inference from `traj_len`/`pred_lens` arithmetic plus the one
corroborating eval config, not a directly-stated fact.

I also fixed an unrelated, unambiguous copy-paste bug while I was in these
files: `train_two_stage_kbos.yaml` and `train_two_stage_klax.yaml` both had
`ckpt: kmsy2` (should be `kbos2` / `klax2` respectively) - i.e. their
`mode_ckpt_path`/`traj_ckpt_path` pointed at KMSY's checkpoint folder. Fixed
to `kbos2` / `klax2`. `train_two_stage_kmsy.yaml` already had the correct
`ckpt: kmsy2`.

The new 20s configs use distinct `ckpt: {airport}_20` identifiers (not
`{airport}2`) specifically so the 20s and 50s checkpoints for the same
airport never land in the same folder.

## 5. Other files touched/added, and things left alone on purpose

- `configs/train_two_stage_20.yaml` and `amelia_tf/train_two_stage_20.py`
  (the old KMSY-only, `paths: default4.yaml`-referencing ad-hoc pair) are
  **left in place, unmodified except for an added deprecation comment** -
  I did not delete them since I can't be certain nothing else references
  them. `paths/default4.yaml` genuinely does not exist under
  `configs/paths/`, so this pair was already non-functional; don't submit it.
  Also note its Stage 2 is architecturally different from the
  kbos/klax/kmsy trainers (it loads and freezes a Stage-1 mode_net for
  conditioning, no teacher forcing) - one more reason not to mix it into this
  grid.
- `num_futures` (1T/2T/4T) is not stored in any yaml; it's set via the Hydra
  CLI override `model.traj_net.config.decoder.num_futures=<1|2|4>` in the
  trajectory-sweep `.sh` scripts, matching how it's always been passed in
  this repo (`configs/model/combined_traj_pred.yaml:123` only has the
  default `num_futures: 1`).
- SLURM headers (`partition=sae`, `account=pilot_sae_gpu`, `gres=gpu:1`,
  module load lines, `conda activate amelia_env`) are copied verbatim from
  `train_kbos.sh`. Mode-only jobs get `--time=48:00:00` (Stage 1 alone, an
  unverified guess - please adjust after seeing real Stage-1 wall-clock
  time); trajectory-sweep jobs keep the existing `120:00:00` budget used
  elsewhere in this repo.
- Did not touch: `AmeliaTF_main`, `AmeliaTF_main_acceleration`,
  `AmeliaTF_main_two_phases`, `STGCNN_baseline`, or anything under
  `AmeliaTF_main_two_phases_4T` unrelated to this grid (e.g. `eval*.yaml`,
  `train.sh`/`train20.sh`, the generic `train_two_stage.py`/`.yaml`).
- No `wandb.login(key=...)` was found in any `train_two_stage_*.py` in this
  folder (unlike the previously-flagged leak in `AmeliaTF_main`), and none
  was added to the new `_20.py` files.

## 6. File inventory

**Code (modified):**
- `amelia_tf/data/datamodule.py` - `use_mode_weights` gate
- `amelia_tf/models/traj_pred_combined.py` - `None`-safe `sample_weights`
- `amelia_tf/train_two_stage_kbos.py`, `train_two_stage_klax.py`,
  `train_two_stage_kmsy.py` - checkpoint-reuse mechanism (`shutil` import,
  `load_pretrained_mode_model`, fixed-path checkpoint copy, `run()` branching)
- `amelia_tf/train_two_stage_20.py` - deprecation docstring only

**Code (new):**
- `amelia_tf/train_two_stage_kbos_20.py`, `train_two_stage_klax_20.py`,
  `train_two_stage_kmsy_20.py`

**Config (modified):** `configs/train_two_stage_kbos.yaml`,
`train_two_stage_klax.yaml`, `train_two_stage_kmsy.yaml` (data/paths horizon
fix, ckpt bug fix, `skip_mode_training`/`skip_traj_training`,
`use_mode_weights: false`); `configs/train_two_stage_20.yaml` (deprecation
comment only)

**Config (new):** `configs/train_two_stage_kbos_20.yaml`,
`train_two_stage_klax_20.yaml`, `train_two_stage_kmsy_20.yaml`

**SLURM scripts (new, 12 files covering 24 job instances):**
`train_mode_{kbos,klax,kmsy}_{50,20}.sh` (6),
`train_traj_{kbos,klax,kmsy}_{50,20}.sh` (6, each submitted 3x with
`NUM_FUTURES={1,2,4}`)
