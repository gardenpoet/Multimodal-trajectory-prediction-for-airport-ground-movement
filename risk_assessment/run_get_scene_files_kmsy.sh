#!/bin/bash
# Same as run_get_scene_files_kbos.sh, for KMSY. See get_scene_files.py's
# module docstring.
#
# Run from the repo root:
#   sbatch risk_assessment/run_get_scene_files_kmsy.sh

#SBATCH --job-name=get_scene_files_kmsy
#SBATCH --output=get_scene_files_kmsy_%j.out
#SBATCH --error=get_scene_files_kmsy_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export HYDRA_FULL_ERROR=1

python -m risk_assessment.get_scene_files \
    data=kmsy.yaml \
    ckpt=kmsy2 \
    +targets_airport=kmsy
