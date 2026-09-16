#!/bin/bash
#SBATCH --job-name=amelia_eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

python -m amelia_tf.eval_two_stage_kbos