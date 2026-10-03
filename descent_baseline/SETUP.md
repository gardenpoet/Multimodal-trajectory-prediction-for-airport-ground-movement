# DESCENT baseline: one-time HPC setup

Retrains DESCENT (Prutsch et al., IROS 2026, https://github.com/a-pru/descent) on
our own runway-filtered scenes instead of its released checkpoints, so the
comparison is apples-to-apples with our retrained STGCNN/Amelia-TF baselines
(their released checkpoints were trained on stock AmeliaScenes output, which
keeps runway-taxiing trajectories that our `scene_processor.py` drops).

Run these steps **interactively** (not via sbatch) once, before submitting any
of the `run_train_descent_*.sh` jobs in this folder.

```bash
cd /gpfs/scratch/exy064/ljx/Risk-Assessment

module load miniforge/25.3.0
module load cuda/12.6.2-gcc-12.2.0   # closest available to the cu128 wheel below; `module avail cuda` showed no 12.8

# 1. Clone the repo (checkpoints + maps are git-lfs; we only need maps/ and
#    splits/ for training, but a plain clone pulls everything -- harmless,
#    just slower. Use GIT_LFS_SKIP_SMUDGE=1 + a scoped `git lfs pull` instead
#    if the full checkpoint download is too slow/large for your quota.)
git lfs install
git clone https://github.com/a-pru/descent.git
cd descent

# 2. Dedicated env (DESCENT pins lightning==2.2.0.post0 / torch>=2.4, deliberately
#    separate from amelia_env to avoid clobbering package versions other jobs rely on)
conda create -n descent_env python=3.9 -y
conda activate descent_env
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# 3. Sanity check: confirm our 3 target airports' maps/splits are present
ls maps/ | grep -E "kmsy|kbos|klax"
ls splits/ | grep -E "kmsy|kbos|klax"
```

No dataset download/symlink step is needed: the training jobs below override
`paths.base_dir` directly to point at our own already-generated, runway-filtered
`proc_full_scenes` directory (`/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia`),
instead of the stock Amelia-10 download the README describes. DESCENT's own
`maps/` (directed-edge airport graphs) and `splits/` (day-based hour-shard
lists) stay untouched -- both are airport-topology/shard-granularity, not
scene-content, so our agent-level runway filtering doesn't invalidate them.

Once the env is set up, submit the three training jobs from the **model repo
root** (so `$SLURM_SUBMIT_DIR` resolves correctly):

```bash
sbatch descent_baseline/run_train_descent_kmsy.sh
sbatch descent_baseline/run_train_descent_kbos.sh
sbatch descent_baseline/run_train_descent_klax.sh
```

Each run trains for up to 100 epochs with early stopping (patience=10 on
`val_ade/t=max`); checkpoints land under
`descent/out/logs/train/runs/<timestamp>/checkpoints/`, and the best one is
auto-evaluated on the test split at the end (same minADE/minFDE@4-modes metric
as the paper's Tables I/II -- still need to reconcile this against our own
MADE/MFDE oracle-vs-scorer convention before it goes in the same table).
