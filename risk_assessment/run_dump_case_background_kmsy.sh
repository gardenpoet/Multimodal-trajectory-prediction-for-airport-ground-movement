#!/bin/bash
# Crops KMSY's real background image (bkg_map.png) to the local area around
# the confirmed case (batch=18, sample=27 -- re-selected 2026-09-27 for
# case-study legibility, see run_case_risk_dynamics_kmsy_1T.sh's comment).
# Pure CPU work, compute partition.
#
# Run from the repo root:
#   sbatch risk_assessment/run_dump_case_background_kmsy.sh

#SBATCH --job-name=dump_case_background_kmsy
#SBATCH --output=risk_assessment/dump_case_background_kmsy_%j.out
#SBATCH --error=risk_assessment/dump_case_background_kmsy_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

OUT_DIR="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out"
mkdir -p "$OUT_DIR"

# Case re-selected 2026-09-27 for case-study legibility (batch=18, sample=27
# -- see run_case_risk_dynamics_kmsy_1T.sh's comment). Bounds computed from
# the new case's own GT extent (ego+ref+other agents, hist+fut) + 250m
# padding.
python -m risk_assessment.dump_case_background \
    --assets_dir /gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia/assets \
    --airport kmsy \
    --north 30.001340 --south 29.987574 --east -90.245605 --west -90.278123 \
    --output_png "$OUT_DIR/kmsy_case_18_27_bg.png" \
    --output_json "$OUT_DIR/kmsy_case_18_27_bg.json"
