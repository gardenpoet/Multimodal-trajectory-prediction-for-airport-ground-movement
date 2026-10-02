#!/bin/bash
# Final stragglers across the b3/s70 and b762/s32 dense 50-frame batches,
# confirmed missing by diffing the downloaded HPC runs/timeline folder
# against the full expected file list (b7/s62 is already 265/265 complete).
# Overwhelmingly STGCNN trajectory jobs, plus a handful of Amelia-TF/dynamics
# cells for b3/s70 -- all use the already-fixed per-frame scene_file, so
# these are resubmits of jobs that simply failed/queued out, not a new bug.
#
# Run from the repo root:
#   bash risk_assessment/retry3_kbos_dense50_stragglers.sh

# --- b3/s70 (43 jobs) ---
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b3s70_f211_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b3s70_f198_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b3s70_f204_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b3s70_f209_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b3s70_f188_4T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b3s70_f209_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f192_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f194_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f203_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f208_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f212_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f217_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f220_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f222_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f232_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f233_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f184_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f187_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f189_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f191_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f195_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f196_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f197_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f198_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f203_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f206_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f207_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f208_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f209_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f211_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f212_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f216_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f217_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f218_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f219_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f220_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f221_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f222_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f223_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f224_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f225_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f226_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f230_stgcnn.sh"

# --- b762/s32 (6 jobs) ---
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1114_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1125_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1128_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1155_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1156_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1157_stgcnn.sh"
