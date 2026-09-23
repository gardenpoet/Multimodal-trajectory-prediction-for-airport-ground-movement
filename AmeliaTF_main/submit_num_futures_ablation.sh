#!/bin/bash
# Batch-submits the 6 Amelia num_futures ablation jobs (KBOS/KLAX/KMSY x
# 8T/16T, H50), each a full 120h retrain from scratch since num_futures
# slices the decoder embedding dim. Run from AmeliaTF_main/, e.g.:
#   bash submit_num_futures_ablation.sh
#
# IMPORTANT before submitting: the override path used in these scripts is
# model.net.config.decoder.num_futures=<K> -- verify it actually reaches
# amelia_tf/models/components/gmm.py's self.num_futures (e.g. via a quick
# `python -m amelia_tf.train_kbos --cfg job model.net.config.decoder.num_futures=8
# | grep num_futures` dry run on the HPC) before committing 6 x 120h jobs to
# the queue, since this repo's local git clone has been found stale relative
# to the HPC on other config paths this session (train_klax.yaml's data/paths
# defaults) and hasn't been independently re-verified for this override.

set -e

SCRIPTS=(
    train_kbos_8T.sh
    train_kbos_16T.sh
    train_klax_8T.sh
    train_klax_16T.sh
    train_kmsy_8T.sh
    train_kmsy_16T.sh
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "WARNING: $script not found in $(pwd), skipping."
        continue
    fi
    jobid=$(sbatch "$script" | awk '{print $NF}')
    echo "Submitted $script -> job $jobid"
done
