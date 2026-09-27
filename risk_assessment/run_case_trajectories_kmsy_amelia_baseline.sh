#!/bin/bash
# Cross-model trajectory dump for KMSY's confirmed candidate
# (scene_file=kmsy/KMSY_455_1689728400/001003_n-5.pkl), AmeliaTF_main
# (plain, non-manoeuvre-conditioned) baseline. Re-selected 2026-09-27 for
# case-study legibility -- see run_case_risk_dynamics_kmsy_1T.sh's comment.
# Scans the FULL test set -- give it ample time.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kmsy_amelia_baseline.sh

#SBATCH --job-name=case_trajectories_kmsy_baseline
#SBATCH --output=risk_assessment/case_trajectories_kmsy_baseline_%j.out
#SBATCH --error=risk_assessment/case_trajectories_kmsy_baseline_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_18_27_trajectories_amelia_baseline.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kmsy \
    +data.dataset.config.random_ego=false \
    +case_scene_file=kmsy/KMSY_455_1689728400/001003_n-5.pkl \
    +output_json="$OUT_JSON"
