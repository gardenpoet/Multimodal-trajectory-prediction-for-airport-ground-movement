#!/bin/bash
# Adds ego_hist_displacement_m / ego_track_straightness to the EXISTING
# klax_50_4T_cases.csv -- no model, no checkpoint, no GPU (see
# rank_candidates.py's docstring). Use this to screen the CSV's other
# candidate rows for a legible replacement case-study trajectory instead of
# re-running the full (GPU) find_cases_two_stage.py pass.
#
# Run from the repo root:
#   sbatch risk_assessment/run_rank_candidates_klax.sh

#SBATCH --job-name=rank_candidates_klax
#SBATCH --output=risk_assessment/rank_candidates_klax_%j.out
#SBATCH --error=risk_assessment/rank_candidates_klax_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

IN_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/klax_50_4T_cases.csv"
OUT_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/klax_50_4T_cases_ranked.csv"

# Same +limit_batches as run_find_cases_klax_50_4T.sh's default, so every
# scene_file already in IN_CSV is guaranteed to be encountered.
python -m risk_assessment.rank_candidates \
    ckpt=klax2 \
    data=klax.yaml \
    +data.dataset.config.random_ego=false \
    data.dataset.config.add_context=false \
    +input_csv="$IN_CSV" \
    +output_csv="$OUT_CSV" \
    +limit_batches=420
