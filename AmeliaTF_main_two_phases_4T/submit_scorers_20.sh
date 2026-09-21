#!/bin/bash
# Batch-submits all six 20s-horizon scorer jobs (KMSY/KBOS/KLAX x 2T/4T),
# each a combined train+test job (12h time limit, scorer.stage=score) --
# see eval_scorer_{kmsy,kbos,klax}_20_{2T,4T}.sh for the per-job details.
#
# Run this from the directory containing those six scripts
# (AmeliaTF_main_two_phases_4T/), e.g.:
#   bash submit_scorers_20.sh
#
# Requires the checkpoints already copied into place at:
#   ${ckpt_dir}/Single-Airport/{kmsy_20,kbos_20,klax_20}/mode_model/{ckpt}_twophases_20.ckpt
#   ${ckpt_dir}/Single-Airport/{kmsy_20,kbos_20,klax_20}/traj_model/{ckpt}_twophases_{2,4}_20.ckpt

set -e

SCRIPTS=(
    eval_scorer_kmsy_20_2T.sh
    eval_scorer_kmsy_20_4T.sh
    eval_scorer_kbos_20_2T.sh
    eval_scorer_kbos_20_4T.sh
    eval_scorer_klax_20_2T.sh
    eval_scorer_klax_20_4T.sh
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "WARNING: $script not found in $(pwd), skipping."
        continue
    fi
    jobid=$(sbatch "$script" | awk '{print $NF}')
    echo "Submitted $script -> job $jobid"
done
