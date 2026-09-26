#!/bin/bash
# One-off diagnostic (see get_scene_files.py's module docstring): fetches
# the stable scene_file identifier for KBOS's 10 case-study candidates, so
# they can be cross-referenced against a *_stgcnn_cases.csv run with the
# same fix, without a full find_cases_two_stage.py rerun. Pure CPU work (no
# model/checkpoint loaded), so targets the CPU-only `compute` partition --
# see run_check_agent_types_kbos_compute.sh for why.
#
# Run from the repo root:
#   sbatch risk_assessment/run_get_scene_files_kbos.sh

#SBATCH --job-name=get_scene_files_kbos
#SBATCH --output=get_scene_files_kbos_%j.out
#SBATCH --error=get_scene_files_kbos_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export HYDRA_FULL_ERROR=1

python -m risk_assessment.get_scene_files \
    data=kbos.yaml \
    ckpt=kbos2 \
    +targets_airport=kbos
