#!/bin/bash
#SBATCH --job-name=amelia_processor     
#SBATCH --output=split_10_%j.out       
#SBATCH --error=split_10_%j.err        

#SBATCH --cpus-per-task=8                
#SBATCH --mem=60G                     
#SBATCH --time=1:00:00               

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python -m amelia_scenes.run_create_splits
