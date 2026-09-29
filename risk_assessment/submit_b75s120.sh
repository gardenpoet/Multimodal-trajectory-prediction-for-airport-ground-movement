#!/bin/bash
# Submits all 5 jobs for the new ambiguity-gated illustrative case
# (batch=75, sample=120) -- see
# run_case_risk_dynamics_kbos_groupb_b75s120_4T.sh for why this one was
# picked (genuinely ambiguous=True, unlike every other case-study pick).
#
# Run from the repo root:
#   bash risk_assessment/submit_b75s120.sh

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b75s120_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b75s120_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b75s120_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_groupb_b75s120_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_groupb_b75s120_amelia_baseline.sh"

echo "Submitted 5 jobs."
