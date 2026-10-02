#!/bin/bash
# Submits the both-sides-predicted jobs for the other two headline cases
# (b3/s70's were submitted separately via submit_kbos_groupb_b3s70_swapego.sh):
#   - confirmed near-miss: batch=7, sample=62, real min sep 22.3m
#   - ambiguous intention: batch=762, sample=32, real min sep 77.4m
# Requires the ego_agent_id fix to be present in AmeliaTF_main,
# AmeliaTF_main_two_phases_4T and STGCNN_baseline's amelia_dataset.py/
# base_dataset.py.
#
# Run from the repo root:
#   bash risk_assessment/submit_kbos_headline_swapego_remaining.sh

# --- confirmed near-miss (b7/s62) ---
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_4T_swapego.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_stgcnn_swapego.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_amelia_baseline_swapego.sh"

# --- ambiguous intention (b762/s32) ---
sbatch "risk_assessment/run_case_risk_dynamics_kbos_762s32_4T_swapego.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_762s32_stgcnn_swapego.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_762s32_amelia_baseline_swapego.sh"
