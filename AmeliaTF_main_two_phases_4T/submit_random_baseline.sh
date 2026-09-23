#!/bin/bash
# Batch-submits the 12 random-candidate-selection baseline test-only runs
# (KMSY/KBOS/KLAX x 2T/4T x H20/H50), for Table 5's oracle-vs-scorer
# comparison. Each reuses already-trained mode+traj checkpoints via
# amelia_tf.eval_two_stage's default "eval" stage (no scorer, no training)
# with model.extra_params.selection_mode=random.
#
# Run this from AmeliaTF_main_two_phases_4T/, e.g.:
#   bash submit_random_baseline.sh

set -e

SCRIPTS=(
    eval_random_kmsy_2T_test.sh
    eval_random_kmsy_4T_test.sh
    eval_random_kmsy_20_2T_test.sh
    eval_random_kmsy_20_4T_test.sh
    eval_random_kbos_2T_test.sh
    eval_random_kbos_4T_test.sh
    eval_random_kbos_20_2T_test.sh
    eval_random_kbos_20_4T_test.sh
    eval_random_klax_2T_test.sh
    eval_random_klax_4T_test.sh
    eval_random_klax_20_2T_test.sh
    eval_random_klax_20_4T_test.sh
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "WARNING: $script not found in $(pwd), skipping."
        continue
    fi
    jobid=$(sbatch "$script" | awk '{print $NF}')
    echo "Submitted $script -> job $jobid"
done
