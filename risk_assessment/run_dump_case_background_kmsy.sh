#!/bin/bash
# Crops KMSY's real background image (bkg_map.png) to the local area around
# the confirmed case (batch=292, sample=46). Pure CPU work, compute
# partition.
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

python -m risk_assessment.dump_case_background \
    --assets_dir /gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia/assets \
    --airport kmsy \
    --north 29.999776 --south 29.985984 --east -90.248805 --west -90.278239 \
    --output_png "$OUT_DIR/kmsy_case_292_46_bg.png" \
    --output_json "$OUT_DIR/kmsy_case_292_46_bg.json"
