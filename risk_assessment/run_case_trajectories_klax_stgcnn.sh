#!/bin/bash
# Cross-model trajectory dump for KLAX's confirmed candidate
# (scene_file=klax/KLAX_455_1684537200/003475_n-8.pkl), STGCNN_baseline.
# Re-selected 2026-09-27 for case-study legibility -- see
# run_case_risk_dynamics_klax_1T.sh's comment. Scans the FULL test set --
# give it ample time.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_klax_stgcnn.sh

#SBATCH --job-name=case_trajectories_klax_stgcnn
#SBATCH --output=risk_assessment/case_trajectories_klax_stgcnn_%j.out
#SBATCH --error=risk_assessment/case_trajectories_klax_stgcnn_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/klax_case_65_41_trajectories_stgcnn.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_stgcnn \
    --config-name=train_stgcnn_klax \
    +data.dataset.config.random_ego=false \
    train=false \
    'ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-23-16/checkpoints/epoch_176.ckpt' \
    +case_scene_file=klax/KLAX_455_1684537200/003475_n-8.pkl \
    +output_json="$OUT_JSON"
