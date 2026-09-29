#!/bin/bash
# Submits all 5 jobs for the new ambiguity-gated illustrative case
# (batch=201, sample=103) -- see
# run_case_risk_dynamics_kbos_groupb_b201s103_4T.sh for why this one was
# picked (genuinely ambiguous=True, unlike every other case-study pick).
#
# Run from the repo root:
#   bash risk_assessment/submit_b201s103.sh

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b201s103_1T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b201s103_2T.sh"
sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_b201s103_4T.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_groupb_b201s103_stgcnn.sh"
sbatch "risk_assessment/run_case_trajectories_kbos_groupb_b201s103_amelia_baseline.sh"

echo "Submitted 5 jobs."
