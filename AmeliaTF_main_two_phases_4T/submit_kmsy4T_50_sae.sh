#!/bin/bash
# Submits the kmsy 50s-horizon trajectory jobs (4T) that never actually
# got submitted from submit_klax4T50_and_kmsy.sh (likely an andrena job/queue
# limit hit partway through that script's loop). Uses the sae partition
# instead, since andrena didn't pick these two up.
#
# Requires train_mode_kmsy_50.sh to have already completed (produces the
# fixed mode checkpoint these jobs reuse via skip_mode_training=true).
#
# Usage:
#   ./submit_kmsy4T_50_sae.sh

set -euo pipefail

PARTITION=sae
ACCOUNT=pilot_sae_gpu

for nf in 4; do
    echo "Submitting train_traj_kmsy_50.sh NUM_FUTURES=$nf on $PARTITION"
    sbatch --partition="$PARTITION" --account="$ACCOUNT" \
        --export=ALL,NUM_FUTURES="$nf" train_traj_kmsy_50.sh
done
