#!/bin/bash
# Batch-submits every Contribution 3 risk-assessment case-finding script
# (two-stage 1T/2T/4T, Amelia baseline, STGCNN baseline) currently in this
# folder, all against KLAX H50.
#
# Run this from the REPO ROOT (the folder containing risk_assessment/ and
# every AmeliaTF_main*/STGCNN_baseline folder as siblings), e.g.:
#   cd /gpfs/scratch/exy064/ljx/Risk-Assessment/
#   bash risk_assessment/submit_all_find_cases.sh
#
# For a first sanity-check pass (a handful of batches each, seconds to
# run), set LIMIT before calling this script -- it's forwarded to every
# job the same way `sbatch --export=ALL,LIMIT=5 <script>` would:
#   LIMIT=5 bash risk_assessment/submit_all_find_cases.sh
# Once every job's .out/.err looks right, resubmit without LIMIT for the
# full runs.

set -e

SCRIPTS=(
    risk_assessment/run_find_cases_klax_50.sh
    risk_assessment/run_find_cases_klax_50_2T.sh
    risk_assessment/run_find_cases_klax_50_4T.sh
    risk_assessment/run_find_cases_klax_50_amelia_baseline.sh
    risk_assessment/run_find_cases_klax_50_stgcnn.sh
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "WARNING: $script not found in $(pwd), skipping."
        continue
    fi
    if [ -n "$LIMIT" ]; then
        jobid=$(sbatch --export=ALL,LIMIT="$LIMIT" "$script" | awk '{print $NF}')
    else
        jobid=$(sbatch "$script" | awk '{print $NF}')
    fi
    echo "Submitted $script -> job $jobid"
done
