#!/bin/bash
# Submits all 18 SLURM jobs for the 6 KBOS group_b candidates picked from
# the bulk group_b screening (case_risk_dynamics 4T + case_trajectories_
# stgcnn + case_trajectories_amelia_baseline, for b3s70/b3s34/b6s112/
# b6s40/b5s9/b5s99).
#
# Run from the repo root:
#   bash risk_assessment/submit_all_groupb_cases.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TAGS=(b3s70 b3s34 b6s112 b6s40 b5s9 b5s99)

for tag in "${TAGS[@]}"; do
    sbatch "risk_assessment/run_case_risk_dynamics_kbos_groupb_${tag}_4T.sh"
    sbatch "risk_assessment/run_case_trajectories_kbos_groupb_${tag}_stgcnn.sh"
    sbatch "risk_assessment/run_case_trajectories_kbos_groupb_${tag}_amelia_baseline.sh"
done

echo "Submitted 18 jobs."
