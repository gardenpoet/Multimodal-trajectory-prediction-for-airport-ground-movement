#!/bin/bash
# Amelia num_futures ablation: retrain KBOS H20 with 8 raw candidate
# trajectories per mode instead of the default 4 (no scorer -- this is the
# plain Amelia decoder, not the two_phases repo's scorer setup). Requires a
# full retrain since num_futures slices the decoder's embedding dim
# (in_size // num_futures), not just a head config change.
#
# H20 needs both data=kbos2.yaml (traj_len=30, drives model.net.config.encoder.T_size)
# and paths=default2.yaml (scenes_dir -> proc_full_scenes2/, the 20s-horizon
# preprocessed data) -- setting only data= is not enough, see the H20 config
# gotcha already hit in the two-stage repo (same underlying paths default).
#
# No task_name override: it's a strict enum (extra_params.task_names:
# [train, eval] in configs/data/default.yaml, asserted in datamodule.py:206),
# not a free-form label -- overriding it crashes data loading. The per-run
# timestamped hydra output dir already keeps this run's checkpoints/ separate
# from kbos_baseline_20's.

#SBATCH --job-name=amelia_train_kbos_8T_20
#SBATCH --output=kbos_8T_20.out
#SBATCH --error=kbos_8T_20.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=120:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

python -m amelia_tf.train_kbos \
    data=kbos2.yaml \
    paths=default2.yaml \
    model.net.config.decoder.num_futures=8
