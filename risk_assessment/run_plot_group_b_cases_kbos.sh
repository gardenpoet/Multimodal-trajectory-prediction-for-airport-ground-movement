#!/bin/bash
# Bulk-plots KBOS's "Group B" candidates -- gt-mode prediction itself safe,
# but SOME OTHER mode/candidate lands inside the safety margin (the
# "risk_worst_case looks beyond the single most-likely prediction" story;
# see risk_assessment.common.select_candidates's criterion="group_b" and
# run_case_risk_dynamics_kmsy_groupb_4T.sh for the original hand-picked
# example this generalizes). Same GT-only, no-model/GPU-needed approach as
# run_plot_risk_cases_kbos.sh.
#
# Sanity-check first with a couple of scenes:
#   sbatch --export=ALL,LIMIT=3 risk_assessment/run_plot_group_b_cases_kbos.sh
# Tune icon size if needed (default 15x amelia_scenes' own tiny default):
#   sbatch --export=ALL,LIMIT=3,ICON_ZOOM=25 risk_assessment/run_plot_group_b_cases_kbos.sh
# Then the full run:
#   sbatch risk_assessment/run_plot_group_b_cases_kbos.sh

#SBATCH --job-name=plot_group_b_cases_kbos
#SBATCH --output=risk_assessment/plot_group_b_cases_kbos_%j.out
#SBATCH --error=risk_assessment/plot_group_b_cases_kbos_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

IN_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_4T_cases_ranked.csv"
OUT_DIR="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/case_screenshots_groupb/kbos"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then LIMIT_ARG="--limit $LIMIT"; fi
ICON_ZOOM_ARG=""
if [ -n "$ICON_ZOOM" ]; then ICON_ZOOM_ARG="--icon_zoom_scale $ICON_ZOOM"; fi

python -m risk_assessment.plot_risk_cases \
    --input_csv "$IN_CSV" \
    --out_dir "$OUT_DIR" \
    --airport kbos \
    --criterion group_b \
    $LIMIT_ARG $ICON_ZOOM_ARG
