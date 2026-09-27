#!/bin/bash
# Crops KLAX's real background image (bkg_map.png) to the local area around
# the confirmed case (batch=317, sample=74). Pure CPU work, compute
# partition.
#
# Run from the repo root:
#   sbatch risk_assessment/run_dump_case_background_klax.sh

#SBATCH --job-name=dump_case_background_klax
#SBATCH --output=risk_assessment/dump_case_background_klax_%j.out
#SBATCH --error=risk_assessment/dump_case_background_klax_%j.err

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
    --airport klax \
    --north 33.950388 --south 33.936180 --east -118.377638 --west -118.419857 \
    --output_png "$OUT_DIR/klax_case_317_74_bg.png" \
    --output_json "$OUT_DIR/klax_case_317_74_bg.json"
