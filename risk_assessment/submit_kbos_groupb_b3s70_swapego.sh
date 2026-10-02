#!/bin/bash
# Submits the 3 both-sides-predicted jobs for the ACTUAL b3/s70 headline
# case used in main.tex (batch=3, sample=70, scene
# kbos/KBOS_183_1673186400/000208_n-9.pkl, real min separation 84.8m).
# Requires the ego_agent_id fix (same commit) to be present in
# AmeliaTF_main, AmeliaTF_main_two_phases_4T and STGCNN_baseline's
# amelia_dataset.py/base_dataset.py.
#
# Run from the repo root:
#   bash risk_assessment/submit_kbos_groupb_b3s70_swapego.sh

sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b3s70_4T_swapego.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_groupb_b3s70_stgcnn_swapego.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_groupb_b3s70_amelia_baseline_swapego.sh"
