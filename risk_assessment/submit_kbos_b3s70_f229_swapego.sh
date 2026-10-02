#!/bin/bash
# Submits the 3 both-sides-predicted jobs for the b3/s70 headline case
# (frame 229, batch=1004/sample=117, scene kbos/KBOS_183_1673186400/000229_n-11.pkl).
# Requires the ego_agent_id fix (this commit) to be present in
# AmeliaTF_main, AmeliaTF_main_two_phases_4T and STGCNN_baseline's
# amelia_dataset.py/base_dataset.py.
#
# Run from the repo root:
#   bash risk_assessment/submit_kbos_b3s70_f229_swapego.sh

sbatch "risk_assessment/run_case_risk_dynamics_kbos_b3s70_f229_4T_swapego.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f229_stgcnn_swapego.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b3s70_f229_amelia_baseline_swapego.sh"
