#!/bin/bash
# Re-runs the 2 group_b candidates whose auto-detected reference agent was
# wrong (b3s34, b5s9 -- their dangerous Hold-mode candidate conflicts with a
# DIFFERENT aircraft than the one the realized trajectory got close to, so
# auto-detection picked the wrong one; see 2026-09-28 notes and each
# script's updated header comment). Now with +case_ref_agent_idx= override.
# 10 jobs (5 each), all CPU.
#
# Run from the repo root:
#   bash risk_assessment/submit_ref_agent_fix.sh

set -euo pipefail
cd "$(dirname "$0")/.."

for tag in b3s34 b5s9; do
    sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_${tag}_1T.sh"
    sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_${tag}_2T.sh"
    sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_${tag}_4T.sh"
    sbatch "risk_assessment/run_case_trajectories_kbos_groupb_${tag}_stgcnn.sh"
    sbatch "risk_assessment/run_case_trajectories_kbos_groupb_${tag}_amelia_baseline.sh"
done

echo "Submitted 10 jobs."
