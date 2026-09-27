#!/bin/bash
# Bulk-plots every KBOS risk-critical candidate (true_min_sep_gt < 50m,
# Aircraft-Aircraft, on-road) as a static PNG using the repo's own existing
# visualization code (amelia_scenes.visualization.scene_viz) -- real
# background map + rotated agent icons, no model/GPU needed. See
# plot_risk_cases.py's docstring. Lets you browse many candidates at once
# and pick your own case-study shortlist, instead of one hand-picked
# candidate at a time.
#
# Sanity-check first with a couple of scenes:
#   sbatch --export=ALL,LIMIT=3 risk_assessment/run_plot_risk_cases_kbos.sh
# Tune icon size if needed (default 15x amelia_scenes' own tiny default):
#   sbatch --export=ALL,LIMIT=3,ICON_ZOOM=25 risk_assessment/run_plot_risk_cases_kbos.sh
# Then the full run:
#   sbatch risk_assessment/run_plot_risk_cases_kbos.sh

#SBATCH --job-name=plot_risk_cases_kbos
#SBATCH --output=risk_assessment/plot_risk_cases_kbos_%j.out
#SBATCH --error=risk_assessment/plot_risk_cases_kbos_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

IN_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_4T_cases_ranked.csv"
OUT_DIR="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/case_screenshots/kbos"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then LIMIT_ARG="--limit $LIMIT"; fi
ICON_ZOOM_ARG=""
if [ -n "$ICON_ZOOM" ]; then ICON_ZOOM_ARG="--icon_zoom_scale $ICON_ZOOM"; fi

python -m risk_assessment.plot_risk_cases \
    --input_csv "$IN_CSV" \
    --out_dir "$OUT_DIR" \
    --airport kbos \
    $LIMIT_ARG $ICON_ZOOM_ARG
