#!/bin/bash
#SBATCH --job-name=create_splits
#SBATCH --output=job.out
#SBATCH --error=job.err

#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

python -m amelia_scenes.run_create_splits