#!/bin/bash
# Retrain DESCENT on KLAX using our own runway-filtered scenes, so the
# comparison is on the same data distribution as our retrained STGCNN/
# Amelia-TF baselines. See descent_baseline/SETUP.md for the one-time env
# setup this depends on.
#
# Run from the repo root:
#   sbatch descent_baseline/run_train_descent_klax.sh

#SBATCH --job-name=train_descent_klax
#SBATCH --output=descent_baseline/train_descent_klax_%j.out
#SBATCH --error=descent_baseline/train_descent_klax_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

cd /gpfs/scratch/exy064/ljx/Risk-Assessment/descent

module load miniforge/25.3.0
conda activate descent_env
module load cuda/12.6.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

python scripts/train.py \
    data=klax \
    paths.base_dir=/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia
