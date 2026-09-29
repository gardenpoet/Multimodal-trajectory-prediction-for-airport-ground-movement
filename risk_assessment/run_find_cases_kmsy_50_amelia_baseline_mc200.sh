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
#   sbatch --export=ALL,LIMIT=5 risk_assessment/run_find_cases_kmsy_50_amelia_baseline_mc200.sh

#SBATCH --job-name=amelia_risk_kmsy_50_baseline_mc200
#SBATCH --output=risk_assessment/risk_kmsy_50_baseline_mc200_%j.out
#SBATCH --error=risk_assessment/risk_kmsy_50_baseline_mc200_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_50_amelia_baseline_mc200_cases.csv"
mkdir -p "$(dirname "$OUT_CSV")"

# Defaults to a ~1/10 random subsample -- see run_find_cases_kmsy_50.sh's
# comment for why a batch-index prefix is valid here. Override with
# --export=ALL,LIMIT=<batches> for a full run or a quick sanity check.
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="+limit_batches=${LIMIT}"

python -m risk_assessment.find_cases_amelia_baseline \
    --config-name=eval_kmsy \
    +data.dataset.config.random_ego=false \
    +risk_method=mc \
    +mc_threshold_ft=200 \
    +mc_samples=100 \
    +output_csv="$OUT_CSV" \
    $LIMIT_ARG
