#!/bin/bash
# Resubmits the STGCNN/Amelia-TF trajectory jobs for b762/s32 with the
# CORRECTED per-frame scene_file (the first generation hardcoded n-5,
# which was wrong for most/all frames in this scenario).
#
# Run from the repo root:
#   bash risk_assessment/retry_scenefile_kbos_b762s32.sh

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1114_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1114_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1115_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1115_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1124_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1124_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1125_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1125_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1126_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1126_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1127_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1127_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1128_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1128_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1154_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1154_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1155_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1155_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1156_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1156_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1157_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1157_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1158_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1158_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1159_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1159_amelia_baseline.sh"

echo "Submitted 26 corrected trajectory jobs."
