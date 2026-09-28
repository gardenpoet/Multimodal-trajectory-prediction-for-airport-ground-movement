#!/bin/bash
# Submits all 40 SLURM jobs for the 8 finalized 2026-09-28 case candidates
# (case_risk_dynamics 1T/2T/4T + case_trajectories_stgcnn +
# case_trajectories_amelia_baseline, for each of KBOS b7s62/b73s51, KLAX
# b71s125/b387s94/b398s59, KMSY b67s80/b292s46/b224s127).
#
# Run from the repo root:
#   bash risk_assessment/submit_all_cases.sh

set -euo pipefail
cd "$(dirname "$0")/.."

TAGS=(
    "kbos b7s62"
    "kbos b73s51"
    "klax b71s125"
    "klax b387s94"
    "klax b398s59"
    "kmsy b67s80"
    "kmsy b292s46"
    "kmsy b224s127"
)

for entry in "${TAGS[@]}"; do
    read -r airport tag <<< "$entry"
    for horizon in 1T 2T 4T; do
        sbatch "risk_assessment/run_case_risk_dynamics_${airport}_${tag}_${horizon}.sh"
    done
    sbatch "risk_assessment/run_case_trajectories_${airport}_${tag}_stgcnn.sh"
    sbatch "risk_assessment/run_case_trajectories_${airport}_${tag}_amelia_baseline.sh"
done

echo "Submitted 40 jobs."
