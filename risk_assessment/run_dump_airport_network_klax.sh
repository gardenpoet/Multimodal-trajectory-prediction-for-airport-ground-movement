#!/bin/bash
# Dumps KLAX's taxiway/runway/hold-line network (lat/lon edges) for the
# case-study map overlay -- see dump_airport_network.py's docstring. Pure
# CPU work (no model), so targets the CPU-only compute partition.
#
# Run from the repo root:
#   sbatch risk_assessment/run_dump_airport_network_klax.sh

#SBATCH --job-name=dump_airport_network_klax
#SBATCH --output=risk_assessment/dump_airport_network_klax_%j.out
#SBATCH --error=risk_assessment/dump_airport_network_klax_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/klax_network.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.dump_airport_network \
    --assets_dir /gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia/assets \
    --airport klax \
    --output_json "$OUT_JSON"
