#!/bin/bash
#SBATCH --job-name=amelia_processor      # ???
#SBATCH --output=processor_%j.out        # ????
#SBATCH --error=processor_%j.err         # ????

#SBATCH --cpus-per-task=4                # ?? CPU ??
#SBATCH --mem=16G                        # ???
#SBATCH --time=48:00:00                  # ?????? (HH:MM:SS)

cd $SLURM_SUBMIT_DIR

# ????
module load miniforge/25.3.0
conda activate amelia_env

# ?????
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ?? Python ??(??? GPU)
python -m amelia_scenes.run_processor
