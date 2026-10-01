#!/bin/bash
# Submits all 25 jobs (5 frames x [1T,2T,4T dynamics + stgcnn,amelia_baseline
# trajectories]) for the continuous-prediction 5x5 trajectory-grid figure,
# advisor-requested: one subplot per (model, current-time frame), all on the
# SAME confirmed near-miss case (KBOS b7/s62, scene kbos/KBOS_701_1675098000),
# frames 2170/2172/2174/2176/2178 (2s apart, spanning the risk-onset window
# identified in the timeliness analysis).
#
# Run from the repo root:
#   bash risk_assessment/submit_kbos_b7s62_continuous.sh

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2170_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2170_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2170_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2170_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2170_amelia_baseline.sh"

sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2172_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2172_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2172_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2172_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2172_amelia_baseline.sh"

sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2174_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2174_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2174_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2174_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2174_amelia_baseline.sh"

sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2176_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2176_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2176_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2176_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2176_amelia_baseline.sh"

sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2178_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2178_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2178_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2178_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_b7s62_f2178_amelia_baseline.sh"

echo "Submitted 25 jobs."
