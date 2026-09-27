#!/bin/bash
# Bulk-plots every KMSY risk-critical candidate WITH the two-stage model's
# per-mode candidate predictions overlaid (see plot_risk_cases_pred.py's
# docstring) -- needs the model/GPU, unlike plot_risk_cases.py's ground-
# truth-only screening. Same checkpoint/score-head config as
# run_find_cases_kmsy_50_4T.sh.
#
# Sanity-check first with a couple of scenes:
#   sbatch --export=ALL,LIMIT=3 risk_assessment/run_plot_risk_cases_pred_kmsy.sh
# Then the full run:
#   sbatch risk_assessment/run_plot_risk_cases_pred_kmsy.sh

#SBATCH --job-name=plot_risk_cases_pred_kmsy
#SBATCH --output=risk_assessment/plot_risk_cases_pred_kmsy_%j.out
#SBATCH --error=risk_assessment/plot_risk_cases_pred_kmsy_%j.err

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

IN_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_50_4T_cases_ranked.csv"
OUT_DIR="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/case_screenshots_pred/kmsy"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then LIMIT_ARG="+limit=$LIMIT"; fi

python -m risk_assessment.plot_risk_cases_pred \
    ckpt=kmsy2 \
    +data.dataset.config.random_ego=false \
    data=kmsy.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard.pt' \
    +input_csv="$IN_CSV" \
    +out_dir="$OUT_DIR" \
    +max_sep_km=0.05 \
    $LIMIT_ARG
