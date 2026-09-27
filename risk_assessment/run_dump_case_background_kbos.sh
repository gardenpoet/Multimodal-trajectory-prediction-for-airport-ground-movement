#!/bin/bash
# Crops KBOS's real background image (bkg_map.png) to the local area around
# the confirmed case (batch=379, sample=45 -- re-selected 2026-09-27 for
# case-study legibility, see run_case_risk_dynamics_kbos_1T.sh's comment)
# -- see dump_case_background.py's docstring. Pure CPU work (no model), so
# targets the CPU-only compute partition. Bounds computed from the new
# case's own GT extent (ego+ref+other agents, hist+fut) + 250m padding.
#
# Run from the repo root:
#   sbatch risk_assessment/run_dump_case_background_kbos.sh

#SBATCH --job-name=dump_case_background_kbos
#SBATCH --output=risk_assessment/dump_case_background_kbos_%j.out
#SBATCH --error=risk_assessment/dump_case_background_kbos_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

OUT_DIR="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out"
mkdir -p "$OUT_DIR"

python -m risk_assessment.dump_case_background \
    --assets_dir /gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia/assets \
    --airport kbos \
    --north 42.370120 --south 42.353352 --east -71.004493 --west -71.023325 \
    --output_png "$OUT_DIR/kbos_case_379_45_bg.png" \
    --output_json "$OUT_DIR/kbos_case_379_45_bg.json"
