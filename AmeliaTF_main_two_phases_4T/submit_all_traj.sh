#!/bin/bash
# Submits every trajectory-stage training job (1T/2T/4T, all airports, both
# horizons): 3 airports x 2 horizons x 3 num_futures = 18 sbatch submissions.
#
# Requires the corresponding train_mode_{airport}_{horizon}.sh to have already
# finished (produces the fixed mode checkpoint each of these jobs reuses via
# skip_mode_training=true) -- run this only after all 6 mode-stage jobs are done.
#
# Usage:
#   ./submit_all_traj.sh

set -euo pipefail

AIRPORTS=(kbos klax kmsy)
HORIZONS=(20 50)
NUM_FUTURES_LIST=(1 2 4)

for airport in "${AIRPORTS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
        script="train_traj_${airport}_${horizon}.sh"
        if [ ! -f "$script" ]; then
            echo "WARNING: $script not found, skipping." >&2
            continue
        fi
        for nf in "${NUM_FUTURES_LIST[@]}"; do
            echo "Submitting $script with NUM_FUTURES=$nf"
            sbatch --export=ALL,NUM_FUTURES="$nf" "$script"
        done
    done
done
