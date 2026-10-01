#!/bin/bash
# Second retry for the b7/s62 dense 50-frame batch -- just the 7 files
# still missing after the first retry (82 -> 7).
#
# Run from the repo root:
#   bash risk_assessment/retry2_kbos_b7s62_dense50.sh

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2193_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2207_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2207_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2207_4T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2209_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2209_4T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2218_4T.sh"

echo "Submitted 7 retry jobs."
