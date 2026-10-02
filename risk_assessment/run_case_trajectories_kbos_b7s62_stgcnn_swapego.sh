#!/bin/bash
# Both-sides-predicted risk comparison, confirmed near-miss headline case
# (batch=7, sample=62, real min separation 22.3m). Same scene/pair as
# run_case_trajectories_kbos_b7s62_stgcnn.sh, but with ego and the
# interactive agent swapped (ego_agent_id=1, ref back to 0) -- requires
# the ego_agent_id fix in STGCNN_baseline's amelia_dataset.py
# (random_ego=false previously ignored ego_agent_id and always used
# agent 0). Scans the FULL test set -- give it ample time.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_b7s62_stgcnn_swapego.sh

#SBATCH --job-name=case_traj_kbos_b7s62_stgcnn_swapego
#SBATCH --output=risk_assessment/case_traj_kbos_b7s62_stgcnn_swapego_%j.out
#SBATCH --error=risk_assessment/case_traj_kbos_b7s62_stgcnn_swapego_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_7_62_trajectories_stgcnn_swapego.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_stgcnn \
    --config-name=train_stgcnn_kbos \
    +data.dataset.config.random_ego=false \
    +data.dataset.config.ego_agent_id=1 \
    train=false \
    'ckpt_path=/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_18-23-16/checkpoints/epoch_185.ckpt' \
    +case_batch_idx=7 +case_sample_idx=62 +case_ref_agent_idx=0 \
    +cases_csv="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_4T_cases_ranked.csv" \
    +output_json="$OUT_JSON"
