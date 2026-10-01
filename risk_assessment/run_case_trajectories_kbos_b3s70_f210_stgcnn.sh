#!/bin/bash
# Dense 50-frame continuous-prediction series (b3/s70 (near-zero-probability alternative)), frame=210 / STGCNN cell.
# scene_file passed directly (not in any ranked CSV); n-suffix confirmed
# to match the two-stage model's own naming for every frame in this window.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_b3s70_f210_stgcnn.sh

#SBATCH --job-name=case_traj_kbos_b3s70_f210_stgcnn
#SBATCH --output=risk_assessment/case_traj_kbos_b3s70_f210_stgcnn_%j.out
#SBATCH --error=risk_assessment/case_traj_kbos_b3s70_f210_stgcnn_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b3s70_f210_trajectories_stgcnn.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_stgcnn \
    --config-name=train_stgcnn_kbos \
    +data.dataset.config.random_ego=false \
    train=false \
    'ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-23-16/checkpoints/epoch_185.ckpt' \
    +case_scene_file=kbos/KBOS_183_1673186400/000210_n-5.pkl \
    +output_json="$OUT_JSON"
