#!/bin/bash
# Adds TP-1T/2T to the both-sides-predicted headline-case jobs (4T +
# STGCNN + Amelia-TF were already submitted via
# submit_kbos_groupb_b3s70_swapego.sh and
# submit_kbos_headline_swapego_remaining.sh) -- completes all 5 models
# for each of the 3 cases, matching the horizon-comparison figures
# (1T/2T/4T) as well as the headline risk table.
#
# Run from the repo root:
#   bash risk_assessment/submit_kbos_headline_swapego_1t2t.sh

# --- b3/s70 (batch=3, sample=70) ---
sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b3s70_1T_swapego.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b3s70_2T_swapego.sh"

# --- confirmed near-miss (batch=7, sample=62) ---
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_1T_swapego.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_2T_swapego.sh"

# --- ambiguous intention / b762s32 (batch=762, sample=32) ---
sbatch "risk_assessment/run_case_risk_dynamics_kbos_762s32_1T_swapego.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_762s32_2T_swapego.sh"
