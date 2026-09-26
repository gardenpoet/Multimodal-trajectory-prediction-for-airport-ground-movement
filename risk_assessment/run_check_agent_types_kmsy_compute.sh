#!/bin/bash
# Same as run_check_agent_types_kmsy.sh, but targets the CPU-only `compute`
# partition instead of `andrena`. See run_check_agent_types_klax_compute.sh
# for the rationale (pure CPU work, no GRES quota on `compute`, avoids
# competing with GPU training jobs for the andrena QOS's gres/gpu=8 cap).
#
# Run from the repo root:
#   sbatch risk_assessment/run_check_agent_types_kmsy_compute.sh

#SBATCH --job-name=check_agent_types_kmsy
#SBATCH --output=check_agent_types_kmsy_%j.out
#SBATCH --error=check_agent_types_kmsy_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export HYDRA_FULL_ERROR=1

python -m risk_assessment.check_agent_types \
    data=kmsy.yaml \
    ckpt=kmsy2 \
    +targets_airport=kmsy
