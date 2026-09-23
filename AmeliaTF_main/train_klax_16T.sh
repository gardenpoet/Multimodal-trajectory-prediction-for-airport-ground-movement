#!/bin/bash
# Amelia num_futures ablation: retrain KLAX H50 with 16 raw candidate
# trajectories per mode instead of the default 4. See train_kbos_8T.sh for
# the full rationale/caveats (same script, num_futures=16 instead of 8).
# Off-road eval is disabled by default now (commit 6227dd1).

#SBATCH --job-name=amelia_train_klax_16T
#SBATCH --output=klax_16T_50.out
#SBATCH --error=klax_16T_50.err

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

python -m amelia_tf.train_klax \
    task_name=train_klax_16T \
    model.net.config.decoder.num_futures=16
