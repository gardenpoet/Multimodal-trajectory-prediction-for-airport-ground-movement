#!/bin/bash
# KMSY H50, STGCNN_baseline (Zhang, Zhong & Mahadevan 2022, unimodal) --
# find_cases_stgcnn.py. Single hypothesis per sample, so
# naive/prob_weighted/worst_case/gated collapse to the same number per row.
#
# risk_assessment/ lives at the REPO ROOT (sibling to AmeliaTF_main,
# AmeliaTF_main_two_phases_4T, STGCNN_baseline, ...) -- submit from the repo
# root (e.g. /gpfs/scratch/exy064/ljx/Risk-Assessment/ on HPC).
#
# Checkpoint path re-verified directly from STGCNN_baseline/train_stgcnn_kmsy.sh
# (epoch_110.ckpt from the 2026-09-17_18-22-39 training run).
#
# Sanity-check first with a few batches:
#   sbatch --export=ALL,LIMIT=5 risk_assessment/run_find_cases_kmsy_50_stgcnn.sh

#SBATCH --job-name=amelia_risk_kmsy_50_stgcnn
#SBATCH --output=risk_kmsy_50_stgcnn_%j.out
#SBATCH --error=risk_kmsy_50_stgcnn_%j.err

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

OUT_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/out/risk_assessment/kmsy_50_stgcnn_cases.csv"
mkdir -p "$(dirname "$OUT_CSV")"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="+limit_batches=$LIMIT"
fi

python -m risk_assessment.find_cases_stgcnn \
    --config-name=train_stgcnn_kmsy \
    train=false \
    'ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-22-39/checkpoints/epoch_110.ckpt' \
    +output_csv="$OUT_CSV" \
    $LIMIT_ARG
