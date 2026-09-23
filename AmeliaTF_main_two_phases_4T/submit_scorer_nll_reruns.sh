#!/bin/bash
# Batch-submits the ten test-only reruns needed to refresh NLL after the
# full-distribution NLL fix (commit 9cee61b) -- KMSY's two (already
# existed: eval_scorer_kmsy_{2T,4T}_test.sh) are NOT included here since
# they were written earlier for a different reason (resuming after a
# time-limit kill); resubmit those two separately if their NLL also needs
# refreshing.
#
# Each job reloads the already-trained, already-saved score head
# (scorer.stage=score_test) and only re-runs test -- much cheaper than a
# full retrain. RMSE/MADE/MFDE/PADE/PFDE are unaffected by the NLL fix and
# won't change; only NLL needs refreshing.
#
# Run this from the directory containing these ten scripts
# (AmeliaTF_main_two_phases_4T/), e.g.:
#   bash submit_scorer_nll_reruns.sh

set -e

SCRIPTS=(
    eval_scorer_kbos_2T_test.sh
    eval_scorer_kbos_4T_test.sh
    eval_scorer_klax_2T_test.sh
    eval_scorer_klax_4T_test.sh
    eval_scorer_kmsy_20_2T_test.sh
    eval_scorer_kmsy_20_4T_test.sh
    eval_scorer_kbos_20_2T_test.sh
    eval_scorer_kbos_20_4T_test.sh
    eval_scorer_klax_20_2T_test.sh
    eval_scorer_klax_20_4T_test.sh
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "WARNING: $script not found in $(pwd), skipping."
        continue
    fi
    jobid=$(sbatch "$script" | awk '{print $NF}')
    echo "Submitted $script -> job $jobid"
done
