#!/bin/bash
# EGO-FIX re-run (dense 50-frame continuous-prediction series, b7/s62 (confirmed near-miss)), frame=2199 / STGCNN cell.
# Pins ego to agent_idx=1; interactive/ref agent left on auto-detect.
# See risk_assessment/EGOFIX_NOTES.md.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_b7s62_f2199_stgcnn_egofix.sh

#SBATCH --job-name=case_traj_kbos_b7s62_f2199_stgcnn_egofix
#SBATCH --output=risk_assessment/case_traj_kbos_b7s62_f2199_stgcnn_egofix_%j.out
#SBATCH --error=risk_assessment/case_traj_kbos_b7s62_f2199_stgcnn_egofix_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b7s62_f2199_trajectories_stgcnn_egofix.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_stgcnn \
    --config-name=train_stgcnn_kbos \
    +data.dataset.config.random_ego=false \
    +data.dataset.config.ego_agent_id=1 \
    train=false \
    'ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-23-16/checkpoints/epoch_185.ckpt' \
    +case_scene_file=kbos/KBOS_701_1675098000/002199_n-5.pkl \
    +output_json="$OUT_JSON"
