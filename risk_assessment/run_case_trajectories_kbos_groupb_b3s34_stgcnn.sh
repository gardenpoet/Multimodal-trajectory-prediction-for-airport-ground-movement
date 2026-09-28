#!/bin/bash
# Cross-model trajectory dump (STGCNN_baseline) for KBOS group_b candidate
# (batch=3, sample=34 in the two-stage model's own dataloader
# ordering) -- see run_case_risk_dynamics_kbos_groupb_b3s34_4T.sh. scene_file
# is resolved from the ranked CSV at run time
# (risk_assessment.common.resolve_case_scene_file), not hardcoded. Scans the
# FULL test set -- give it ample time.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_groupb_b3s34_stgcnn.sh

#SBATCH --job-name=case_trajectories_kbos_groupb_b3s34_stgcnn
#SBATCH --output=risk_assessment/case_trajectories_kbos_groupb_b3s34_stgcnn_%j.out
#SBATCH --error=risk_assessment/case_trajectories_kbos_groupb_b3s34_stgcnn_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_3_34_groupb_trajectories_stgcnn.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_stgcnn \
    --config-name=train_stgcnn_kbos \
    +data.dataset.config.random_ego=false \
    train=false \
    'ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-23-16/checkpoints/epoch_185.ckpt' \
    +case_batch_idx=3 +case_sample_idx=34 \
    +cases_csv="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_4T_cases_ranked.csv" \
    +output_json="$OUT_JSON"
