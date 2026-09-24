#!/bin/bash
# First test of find_cases.py -- KLAX H50, 1T checkpoint (single trajectory
# candidate per mode, sidestepping Contribution 2's K-selection question so
# this experiment isolates Contribution 3's MODE-level phenomenon).
#
# This script (and find_cases.py itself) has NOT been executed anywhere
# yet. Submit from the repo root (AmeliaTF_main_two_phases_4T/, same as
# every other .sh script here), not from inside risk_assessment/. Run with
# LIMIT=5 first (a handful of batches, seconds to run) to confirm it works
# end-to-end and to eyeball the output CSV's columns before committing to
# a full run:
#   sbatch --export=ALL,LIMIT=5 risk_assessment/run_find_cases_klax_50.sh
# Once that looks right, submit the full run:
#   sbatch risk_assessment/run_find_cases_klax_50.sh

#SBATCH --job-name=amelia_risk_klax_50
#SBATCH --output=risk_klax_50_%j.out
#SBATCH --error=risk_klax_50_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/risk_assessment/klax_50_cases.csv"
mkdir -p "$(dirname "$OUT_CSV")"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="+limit_batches=$LIMIT"
fi

python -m risk_assessment.find_cases \
    ckpt=klax2 \
    data=klax.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +output_csv="$OUT_CSV" \
    $LIMIT_ARG
