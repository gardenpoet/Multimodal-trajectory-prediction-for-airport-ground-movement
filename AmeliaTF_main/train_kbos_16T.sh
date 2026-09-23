#!/bin/bash
# Amelia num_futures ablation: retrain KBOS H50 with 16 raw candidate
# trajectories per mode instead of the default 4. See train_kbos_8T.sh for
# the full rationale/caveats (same script, num_futures=16 instead of 8).

#SBATCH --job-name=amelia_train_kbos_16T
#SBATCH --output=kbos_16T_50.out
#SBATCH --error=kbos_16T_50.err

#SBATCH --partition=sae
#SBATCH --account=pilot_sae_gpu
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
    task_name=train_kbos_16T \
    model.net.config.decoder.num_futures=16
