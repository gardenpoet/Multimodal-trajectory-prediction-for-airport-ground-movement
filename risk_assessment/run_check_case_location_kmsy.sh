#!/bin/bash
# One-off diagnostic (see check_case_location.py's module docstring): for
# the KMSY case-study candidate whose closest-approach agent type is
# ambiguous ("Unknown"), checks whether the closest-approach point sits on
# or off the airport's taxiway/runway movement-area network -- corroborating
# (not conclusive) evidence for whether this is a routine gate/stand
# encounter or a genuine movement-area near-miss. Pure CPU work (loads the
# datamodule + the semantic-graph reference network, no model/checkpoint),
# so targets the CPU-only `compute` partition -- see
# run_check_agent_types_kmsy_compute.sh for why.
#
# Run from the repo root:
#   sbatch risk_assessment/run_check_case_location_kmsy.sh

#SBATCH --job-name=check_case_location_kmsy
#SBATCH --output=check_case_location_kmsy_%j.out
#SBATCH --error=check_case_location_kmsy_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export HYDRA_FULL_ERROR=1

python -m risk_assessment.check_case_location \
    data=kmsy.yaml \
    ckpt=kmsy2 \
    +targets_airport=kmsy
