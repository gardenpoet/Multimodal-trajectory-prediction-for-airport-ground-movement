#!/bin/bash
# Submits all 5 jobs for the replacement illustrative case (batch=762,
# sample=32) -- see run_case_risk_dynamics_kbos_762s32_4T.sh for why
# this one was picked (replaces the now-invalid b300/s35 case, whose
# gated value collapsed to naive under the 2026-09-30 chain-restricted
# ambiguous-group redefinition; this one's gated=1.0 survives it by
# construction).
#
# Run from the repo root:
#   bash risk_assessment/submit_kbos_762s32.sh

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch "risk_assessment/run_case_risk_dynamics_kbos_762s32_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_762s32_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_762s32_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_762s32_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_762s32_amelia_baseline.sh"

echo "Submitted 5 jobs."
