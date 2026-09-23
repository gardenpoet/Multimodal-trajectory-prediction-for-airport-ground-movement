#!/bin/bash
# Test-only rerun of the already-trained klax_baseline_50 checkpoint, now that
# off-road evaluation is disabled by default (commit 6227dd1) -- avoids the
# slow test hang without needing to redo the 120h training.
#
# Uses amelia_tf/eval.py (generic test-only entrypoint: instantiates model +
# datamodule, then trainer.test(ckpt_path=...) -- no trainer.fit call at all),
# with configs/eval_klax.yaml providing the same data/model/paths composition
# as train_klax.yaml so the test set matches exactly.
#
# ckpt_path in eval_klax.yaml is a best-effort reconstruction of the training
# run's checkpoint dir from the timestamp printed in the klax log's wandb
# notes field (2026-09-18_16-13-14) plus the hydra run-dir pattern -- verify
# it exists on the cluster before submitting and override ckpt_path=... on
# the command line below if the timestamp or filename doesn't match.

#SBATCH --job-name=amelia_eval_klax50_test
#SBATCH --output=eval_klax_50_test_%j.out
#SBATCH --error=eval_klax_50_test_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

python -m amelia_tf.eval --config-name=eval_klax
