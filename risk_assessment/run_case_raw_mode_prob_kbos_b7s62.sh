#!/bin/bash
# One-off diagnostic (2026-09-30): recover the confirmed near-miss case's
# (KBOS batch=7, sample=62) raw, pre-feasibility-mask mode probability for
# the turn currently marked infeasible, by flipping AmeliaMode's own
# apply_hard_mask attribute off for one forward pass -- see
# case_raw_mode_prob.py's docstring. Read-only: no file changes, no
# retraining, prints straight to this job's .out log.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_raw_mode_prob_kbos_b7s62.sh

#SBATCH --job-name=case_raw_mode_prob_kbos_b7s62
#SBATCH --output=risk_assessment/case_raw_mode_prob_kbos_b7s62_%j.out
#SBATCH --error=risk_assessment/case_raw_mode_prob_kbos_b7s62_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

python -m risk_assessment.case_raw_mode_prob \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +case_batch_idx=7 +case_sample_idx=62
