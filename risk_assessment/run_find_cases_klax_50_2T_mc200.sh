#!/bin/bash
# KLAX H50, 2T checkpoint -- find_cases_two_stage.py's mode x candidate risk
# aggregation using the trained scorer's per-candidate probabilities. Same
# checkpoint/score-head config as
# AmeliaTF_main_two_phases_4T/eval_scorer_klax_2T_test.sh.
#
# risk_assessment/ lives at the REPO ROOT (sibling to AmeliaTF_main,
# AmeliaTF_main_two_phases_4T, STGCNN_baseline, ...) -- submit from the repo
# root (e.g. /gpfs/scratch/exy064/ljx/Risk-Assessment/ on HPC).
#
# Sanity-check first with a few batches:
#   sbatch --export=ALL,LIMIT=5 risk_assessment/run_find_cases_klax_50_2T_mc200.sh
# Then the full run:
#   sbatch risk_assessment/run_find_cases_klax_50_2T_mc200.sh

#SBATCH --job-name=amelia_risk_klax_50_2T_mc200
#SBATCH --output=risk_assessment/risk_klax_50_2T_mc200_%j.out
#SBATCH --error=risk_assessment/risk_klax_50_2T_mc200_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=36:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/klax_50_2T_mc200_cases.csv"
mkdir -p "$(dirname "$OUT_CSV")"

# Defaults to a ~1/10 random subsample -- see run_find_cases_klax_50.sh's
# comment for why a batch-index prefix is valid here. Override with
# --export=ALL,LIMIT=<batches> for a full run or a quick sanity check.
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="+limit_batches=${LIMIT}"

python -m risk_assessment.find_cases_two_stage \
    ckpt=klax2 \
    +data.dataset.config.random_ego=false \
    data=klax.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_2_50.ckpt' \
    model.traj_net.config.num_hypotheses=2 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard_2T.pt' \
    +risk_method=mc \
    +mc_threshold_ft=200 \
    +mc_samples=100 \
    +output_csv="$OUT_CSV" \
    $LIMIT_ARG
