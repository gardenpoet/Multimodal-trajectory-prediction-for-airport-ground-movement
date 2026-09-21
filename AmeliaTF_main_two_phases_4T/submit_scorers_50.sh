#!/bin/bash
# Batch-submits the four 50s-horizon KBOS/KLAX scorer jobs (2T/4T each),
# each a combined train+test job (12h time limit, scorer.stage=score) --
# see eval_scorer_{kbos,klax}_{2T,4T}.sh for the per-job details.
# KMSY's 50s scorer already ran separately (eval_scorer_kmsy_{2T,4T}.sh
# plus their _test.sh companions), so it is not included here.
#
# Run this from the directory containing those four scripts
# (AmeliaTF_main_two_phases_4T/), e.g.:
#   bash submit_scorers_50.sh
#
# Requires the checkpoints already copied into place at:
#   ${ckpt_dir}/Single-Airport/{kbos2,klax2}/mode_model/{ckpt}_twophases_50.ckpt
#   ${ckpt_dir}/Single-Airport/{kbos2,klax2}/traj_model/{ckpt}_twophases_{2,4}_50.ckpt

set -e

SCRIPTS=(
    eval_scorer_kbos_2T.sh
    eval_scorer_kbos_4T.sh
    eval_scorer_klax_2T.sh
    eval_scorer_klax_4T.sh
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "WARNING: $script not found in $(pwd), skipping."
        continue
    fi
    jobid=$(sbatch "$script" | awk '{print $NF}')
    echo "Submitted $script -> job $jobid"
done
