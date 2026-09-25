#!/bin/bash
# One-off diagnostic (see check_agent_types.py's module docstring): checks
# whether the ground-truth closest-approach agent in each KBOS case-study
# candidate is an Aircraft or a Vehicle. No model/checkpoint is actually
# needed (pure CPU work), but --gres=gpu:1 is requested anyway since this
# partition/account requires it to submit at all.
#
# Split out from the original combined run_check_agent_types.sh (which
# looped kbos/klax/kmsy sequentially inside a single 1h job): reproducible
# ego-agent selection forces num_workers=0 (single-threaded dataloading), so
# each airport alone already takes 25-45 min just to iterate up to its
# highest target batch_idx -- three airports back-to-back in one 1h job
# doesn't fit (confirmed: job 28411508 was killed mid-KLAX with zero KMSY
# output). Running one airport per job, in parallel, fixes this.
#
# Run from the repo root:
#   sbatch risk_assessment/run_check_agent_types_kbos.sh

#SBATCH --job-name=check_agent_types_kbos
#SBATCH --output=check_agent_types_kbos_%j.out
#SBATCH --error=check_agent_types_kbos_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export HYDRA_FULL_ERROR=1

python -m risk_assessment.check_agent_types \
    data=kbos.yaml \
    ckpt=kbos2 \
    +targets_airport=kbos
