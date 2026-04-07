#!/bin/bash
#SBATCH --job-name=amelia_eval
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

#SBATCH --partition=sae
#SBATCH --account=pilot_sae_gpu

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m amelia_tf.eval