#!/bin/bash
# Re-submits every currently-relevant case-study job (KBOS confirmed b7/s62
# + the 6 group_b candidates, each now getting the full 1T/2T/4T + STGCNN +
# Amelia-baseline treatment) now that case_risk_dynamics.py/
# case_trajectories_stgcnn.py/case_trajectories_amelia_baseline.py also
# capture sigma_xy (+ start_heading_deg) in their output, and now that none
# of these need GPU. 5 (b7/s62) + 6*5 (group_b) = 35 jobs, all CPU.
#
# Run from the repo root:
#   bash risk_assessment/submit_all_cases_with_sigma.sh

set -euo pipefail
cd "$(dirname "$0")/.."

sbatch risk_assessment/run_case_risk_dynamics_kbos_b7s62_1T.sh
sbatch risk_assessment/run_case_risk_dynamics_kbos_b7s62_2T.sh
sbatch risk_assessment/run_case_risk_dynamics_kbos_b7s62_4T.sh
sbatch risk_assessment/run_case_trajectories_kbos_b7s62_stgcnn.sh
sbatch risk_assessment/run_case_trajectories_kbos_b7s62_amelia_baseline.sh

TAGS=(b3s70 b3s34 b6s112 b6s40 b5s9 b5s99)
for tag in "${TAGS[@]}"; do
    sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_${tag}_1T.sh"
    sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_${tag}_2T.sh"
    sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_${tag}_4T.sh"
    sbatch "risk_assessment/run_case_trajectories_kbos_groupb_${tag}_stgcnn.sh"
    sbatch "risk_assessment/run_case_trajectories_kbos_groupb_${tag}_amelia_baseline.sh"
done

echo "Submitted 35 jobs."
