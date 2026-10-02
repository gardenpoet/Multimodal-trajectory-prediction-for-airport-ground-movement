#!/bin/bash
# Final 10 stragglers left after retry3 (b7/s62 is complete; this is just
# b3/s70 + b762/s32). Same jobs as before, still failing with a dataset
# scanning error (self.scenario_list empty) rather than a scene_file bug --
# looks like contention when many STGCNN jobs start at once, so plain
# resubmission is the fix, not a code change.
#
# Run from the repo root:
#   bash risk_assessment/retry4_kbos_dense50_final10.sh

# --- b3/s70 (5 jobs) ---
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f203_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f217_amelia_baseline.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f223_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f226_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f230_stgcnn.sh"

# --- b762/s32 (5 jobs) ---
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1114_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1125_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1128_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1156_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b762s32_f1157_stgcnn.sh"
