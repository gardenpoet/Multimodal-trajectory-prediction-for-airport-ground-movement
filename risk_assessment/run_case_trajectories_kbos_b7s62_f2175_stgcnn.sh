#!/bin/bash
# Fills the one genuine gap (frame 2175) in the b7/s62 dense series. STGCNN
# cell. scene_file n-suffix confirmed to match the two-stage model's own
# naming for this frame (n-5).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_b7s62_f2175_stgcnn.sh

#SBATCH --job-name=case_traj_kbos_b7s62_f2175_stgcnn
#SBATCH --output=risk_assessment/case_traj_kbos_b7s62_f2175_stgcnn_%j.out
#SBATCH --error=risk_assessment/case_traj_kbos_b7s62_f2175_stgcnn_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b7s62_f2175_trajectories_stgcnn.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_stgcnn \
    --config-name=train_stgcnn_kbos \
    +data.dataset.config.random_ego=false \
    train=false \
    'ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-23-16/checkpoints/epoch_185.ckpt' \
    +case_scene_file=kbos/KBOS_701_1675098000/000175_n-5.pkl \
    +output_json="$OUT_JSON"
