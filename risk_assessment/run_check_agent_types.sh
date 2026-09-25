#!/bin/bash
# One-off diagnostic (see check_agent_types.py's module docstring): checks
# whether the ground-truth closest-approach agent in each case-study
# candidate is an Aircraft or a Vehicle (ground service vehicles legitimately
# operate within sub-metre distance of a parked aircraft -- if that's what's
# behind a "near-miss" candidate, it isn't the aircraft-aircraft near miss
# the case study wants). No model/checkpoint needed, so no GPU requested.
#
# Run from the repo root:
#   sbatch risk_assessment/run_check_agent_types.sh

#SBATCH --job-name=check_agent_types
#SBATCH --output=check_agent_types_%j.out
#SBATCH --error=check_agent_types_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export HYDRA_FULL_ERROR=1

for airport in kbos klax kmsy; do
    echo "===== $airport ====="
    python -m risk_assessment.check_agent_types \
        data="${airport}.yaml" \
        ckpt="${airport}2" \
        +targets_airport="$airport"
done
