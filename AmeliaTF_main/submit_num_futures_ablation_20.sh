#!/bin/bash
# Batch-submits the 6 Amelia num_futures ablation jobs at H20 (KBOS/KLAX/KMSY
# x 8T/16T), mirroring submit_num_futures_ablation.sh's H50 batch. Each is a
# full 120h retrain from scratch since num_futures slices the decoder
# embedding dim. Run from AmeliaTF_main/, e.g.:
#   bash submit_num_futures_ablation_20.sh

set -e

SCRIPTS=(
    train_kbos_8T_20.sh
    train_kbos_16T_20.sh
    train_klax_8T_20.sh
    train_klax_16T_20.sh
    train_kmsy_8T_20.sh
    train_kmsy_16T_20.sh
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "WARNING: $script not found in $(pwd), skipping."
        continue
    fi
    jobid=$(sbatch "$script" | awk '{print $NF}')
    echo "Submitted $script -> job $jobid"
done
