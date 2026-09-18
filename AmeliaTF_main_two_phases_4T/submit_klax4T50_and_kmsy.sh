#!/bin/bash
# Submits exactly 7 trajectory-stage jobs on the andrena partition:
#   - klax 4T @ 50s (resubmitted here because it was stuck queued on sae
#     and never started running)
#   - kmsy 1T/2T/4T @ 20s and 50s (6 jobs; kmsy's fixed-num_hypotheses
#     sweep hasn't been submitted at all yet, unlike kbos/klax)
#
# Requires train_mode_kmsy_{20,50}.sh and train_mode_klax_50.sh to have
# already completed (produces the fixed mode checkpoint each of these
# jobs reuses via skip_mode_training=true).
#
# Usage:
#   ./submit_klax4T50_and_kmsy.sh

set -euo pipefail

PARTITION=andrena
ACCOUNT=pilot_andrena

echo "Submitting klax_traj_50 NUM_FUTURES=4 on $PARTITION"
sbatch --partition="$PARTITION" --account="$ACCOUNT" \
    --export=ALL,NUM_FUTURES=4 train_traj_klax_50.sh

for horizon in 20 50; do
    for nf in 1 2 4; do
        script="train_traj_kmsy_${horizon}.sh"
        echo "Submitting $script NUM_FUTURES=$nf on $PARTITION"
        sbatch --partition="$PARTITION" --account="$ACCOUNT" \
            --export=ALL,NUM_FUTURES="$nf" "$script"
    done
done
