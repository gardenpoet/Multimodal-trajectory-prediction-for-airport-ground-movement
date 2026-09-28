#!/bin/bash
# Dumps KBOS's FULL background image (bkg_map.png), uncropped -- replaces
# the narrow crop kbos_bg.jpg/.json currently used (which was scoped to
# just the original confirmed case's local area and is missing background
# for group_b candidates at other locations on the airport). Passing bounds
# wider than the image's own extent makes dump_case_background.py clamp
# down to the image's real (full) extent instead of cropping further.
# Pure CPU work (no model).
#
# Run from the repo root:
#   sbatch risk_assessment/run_dump_case_background_kbos_full.sh

#SBATCH --job-name=dump_case_background_kbos_full
#SBATCH --output=risk_assessment/dump_case_background_kbos_full_%j.out
#SBATCH --error=risk_assessment/dump_case_background_kbos_full_%j.err

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
    --north 90 --south -90 --east 180 --west -180 \
    --output_png "$OUT_DIR/kbos_bg_full.png" \
    --output_json "$OUT_DIR/kbos_bg_full.json"
