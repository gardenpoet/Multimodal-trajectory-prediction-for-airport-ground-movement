#!/bin/bash
# Batch-submits every Contribution 3 risk-assessment case-finding script
# (two-stage 1T/2T/4T, Amelia baseline, STGCNN baseline) currently in this
# folder, across all three airports (KBOS/KLAX/KMSY) at H50. KBOS/KMSY's 1T
# and Amelia-baseline legs are less battle-tested than KLAX's -- see each
# script's own header comment for what's unverified.
#
# Run this from the REPO ROOT (the folder containing risk_assessment/ and
# every AmeliaTF_main*/STGCNN_baseline folder as siblings), e.g.:
#   cd /gpfs/scratch/exy064/ljx/Risk-Assessment/
#   bash risk_assessment/submit_all_find_cases.sh
#
# Each script now defaults to LIMIT unset -> a ~1/10 random subsample
# (batch-index prefix into a seed-shuffled file list, see each script's own
# comment) rather than the full test set -- this is now the standard run,
# not just a sanity check. For a quick syntax/wiring sanity check (a
# handful of batches, seconds to run), set LIMIT before calling this
# script -- it's forwarded to every job the same way
# `sbatch --export=ALL,LIMIT=5 <script>` would:
#   LIMIT=5 bash risk_assessment/submit_all_find_cases.sh
# For a genuine FULL run instead of the ~1/10 default, pass a LIMIT larger
# than the airport's total batch count (see each script's own comment for
# the approximate total).

set -e

SCRIPTS=(
    risk_assessment/run_find_cases_kbos_50.sh
    risk_assessment/run_find_cases_kbos_50_2T.sh
    risk_assessment/run_find_cases_kbos_50_4T.sh
    risk_assessment/run_find_cases_kbos_50_amelia_baseline.sh
    risk_assessment/run_find_cases_kbos_50_stgcnn.sh
    risk_assessment/run_find_cases_klax_50.sh
    risk_assessment/run_find_cases_klax_50_2T.sh
    risk_assessment/run_find_cases_klax_50_4T.sh
    risk_assessment/run_find_cases_klax_50_amelia_baseline.sh
    risk_assessment/run_find_cases_klax_50_stgcnn.sh
    risk_assessment/run_find_cases_kmsy_50.sh
    risk_assessment/run_find_cases_kmsy_50_2T.sh
    risk_assessment/run_find_cases_kmsy_50_4T.sh
    risk_assessment/run_find_cases_kmsy_50_amelia_baseline.sh
    risk_assessment/run_find_cases_kmsy_50_stgcnn.sh
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
