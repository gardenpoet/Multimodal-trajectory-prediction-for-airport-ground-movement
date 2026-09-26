#!/bin/bash
# Same as run_check_agent_types_klax.sh, but targets the CPU-only `compute`
# partition instead of `andrena`. This job is pure CPU work (no
# model/checkpoint loaded), so it doesn't need a GPU at all -- the only
# reason the andrena/pilot_andrena scripts request --gres=gpu:1 is that
# partition's own submission policy, not any real need. `compute` has no
# GRES quota (sacctmgr: MaxTRESPerUser only lists cpu=2000, no gres/gpu
# entry) so this doesn't compete with GPU training jobs for the
# andrena QOS's gres/gpu=8 per-user cap. No --account override: pilot_andrena
# is the pilot allocation tied to andrena/apini/sae, unlikely to be valid on
# compute -- try the default account first.
#
# Run from the repo root:
#   sbatch risk_assessment/run_check_agent_types_klax_compute.sh

#SBATCH --job-name=check_agent_types_klax
#SBATCH --output=check_agent_types_klax_%j.out
#SBATCH --error=check_agent_types_klax_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export HYDRA_FULL_ERROR=1

python -m risk_assessment.check_agent_types \
    data=klax.yaml \
    ckpt=klax2 \
    +targets_airport=klax
