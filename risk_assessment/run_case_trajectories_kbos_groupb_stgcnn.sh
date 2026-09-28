#!/bin/bash
# Cross-model trajectory dump (STGCNN_baseline) for KBOS's Group B
# illustrative case (batch=19, sample=14, scene_file=kbos/KBOS_446_1674144000/
# 001336_n-10.pkl -- see run_case_risk_dynamics_kmsy_groupb_4T.sh's docstring
# for what this case demonstrates: risk_worst_case looking beyond the
# single most-likely prediction). Scans the FULL test set -- give it ample
# time.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_groupb_stgcnn.sh

#SBATCH --job-name=case_trajectories_kbos_groupb_stgcnn
#SBATCH --output=risk_assessment/case_trajectories_kbos_groupb_stgcnn_%j.out
#SBATCH --error=risk_assessment/case_trajectories_kbos_groupb_stgcnn_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_19_14_groupb_trajectories_stgcnn.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_stgcnn \
    --config-name=train_stgcnn_kbos \
    +data.dataset.config.random_ego=false \
    train=false \
    'ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-23-16/checkpoints/epoch_185.ckpt' \
    +case_scene_file=kbos/KBOS_446_1674144000/001336_n-10.pkl \
    +output_json="$OUT_JSON"
