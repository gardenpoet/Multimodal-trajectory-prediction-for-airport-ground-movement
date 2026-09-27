#!/bin/bash
# Crops KBOS's real background image (bkg_map.png) to the local area around
# the confirmed case (batch=103, sample=47) -- see dump_case_background.py's
# docstring. Pure CPU work (no model), so targets the CPU-only compute
# partition.
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
    --north 42.375110 --south 42.357192 --east -71.008830 --west -71.021086 \
    --output_png "$OUT_DIR/kbos_case_103_47_bg.png" \
    --output_json "$OUT_DIR/kbos_case_103_47_bg.json"
