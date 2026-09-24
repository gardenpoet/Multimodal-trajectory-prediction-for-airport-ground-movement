#!/bin/bash
# KMSY H50, AmeliaTF_main (plain, non-manoeuvre-conditioned) baseline --
# find_cases_amelia_baseline.py's flat multi-hypothesis risk aggregation.
# Uses configs/eval_kmsy.yaml (checkpoint consolidated at
# datasets/amelia/checkpoints/Single-Airport/kmsy/kmsy_baseline_50.ckpt).
#
# risk_assessment/ lives at the REPO ROOT (sibling to AmeliaTF_main,
# AmeliaTF_main_two_phases_4T, STGCNN_baseline, ...) -- submit from the repo
# root (e.g. /gpfs/scratch/exy064/ljx/Risk-Assessment/ on HPC).
#
# Sanity-check first with a few batches:
#   sbatch --export=ALL,LIMIT=5 risk_assessment/run_find_cases_kmsy_50_amelia_baseline.sh

#SBATCH --job-name=amelia_risk_kmsy_50_baseline
#SBATCH --output=risk_kmsy_50_baseline_%j.out
#SBATCH --error=risk_kmsy_50_baseline_%j.err

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

OUT_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/out/risk_assessment/kmsy_50_amelia_baseline_cases.csv"
mkdir -p "$(dirname "$OUT_CSV")"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="+limit_batches=$LIMIT"
fi

python -m risk_assessment.find_cases_amelia_baseline \
    --config-name=eval_kmsy \
    +output_csv="$OUT_CSV" \
    $LIMIT_ARG
