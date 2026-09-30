#!/bin/bash
# Batch-submits all 15 MC(200ft)-based find_cases_*.py jobs (two-stage
# 2T/4T, Amelia baseline, STGCNN baseline, plus the original/1T variant)
# across KBOS/KLAX/KMSY at H50 -- the full-test-set rerun behind
# tab:risk_aggregate, needed again after the 2026-09-30 redefinition of
# the ambiguity-gated strategy's search pool (chain-connected ambiguous
# modes only, not every feasible one) and the removal of the
# has_nearby_agent relevance pre-filter (every sample kept, flagged via
# all_zero_risk instead of silently dropped).
#
# Unlike submit_all_find_cases.sh (the older deterministic-margin
# scripts, which default to a ~1/10 subsample unless LIMIT is set), every
# _mc200 script below defaults to NO limit at all when LIMIT is unset --
# see each script's own LIMIT_ARG line. Do not set LIMIT for a genuine
# full run; the 2026-09-29 run that produced the current
# Downloads/RS_MC/*.csv files did not set it either (row counts there
# already match a full test set, not a 1/10 slice).
#
# Run this from the REPO ROOT (the folder containing risk_assessment/ and
# every AmeliaTF_main*/STGCNN_baseline folder as siblings), e.g.:
#   cd /gpfs/scratch/exy064/ljx/Risk-Assessment/
#   bash risk_assessment/submit_all_find_cases_mc200.sh
#
# For a quick syntax/wiring sanity check only (a handful of batches,
# seconds to run), set LIMIT before calling this script -- it's
# forwarded to every job the same way `sbatch --export=ALL,LIMIT=5
# <script>` would:
#   LIMIT=5 bash risk_assessment/submit_all_find_cases_mc200.sh

set -e

SCRIPTS=(
    risk_assessment/run_find_cases_kbos_50_mc200.sh
    risk_assessment/run_find_cases_kbos_50_2T_mc200.sh
    risk_assessment/run_find_cases_kbos_50_4T_mc200.sh
    risk_assessment/run_find_cases_kbos_50_amelia_baseline_mc200.sh
    risk_assessment/run_find_cases_kbos_50_stgcnn_mc200.sh
    risk_assessment/run_find_cases_klax_50_mc200.sh
    risk_assessment/run_find_cases_klax_50_2T_mc200.sh
    risk_assessment/run_find_cases_klax_50_4T_mc200.sh
    risk_assessment/run_find_cases_klax_50_amelia_baseline_mc200.sh
    risk_assessment/run_find_cases_klax_50_stgcnn_mc200.sh
    risk_assessment/run_find_cases_kmsy_50_mc200.sh
    risk_assessment/run_find_cases_kmsy_50_2T_mc200.sh
    risk_assessment/run_find_cases_kmsy_50_4T_mc200.sh
    risk_assessment/run_find_cases_kmsy_50_amelia_baseline_mc200.sh
    risk_assessment/run_find_cases_kmsy_50_stgcnn_mc200.sh
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
