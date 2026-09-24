#!/bin/bash
# Test-only rerun of the already-trained kbos_baseline_50 checkpoint (same
# 2026-09-18 batch as klax_baseline_50/kmsy_baseline_50), now consolidated
# into datasets/amelia/checkpoints/Single-Airport/kbos/kbos_baseline_50.ckpt
# alongside the two-stage model's kbos2/kbos_20 checkpoints, per
# configs/eval_kbos.yaml.
#
# Uses amelia_tf/eval.py (generic test-only entrypoint: instantiates model +
# datamodule, then trainer.test(ckpt_path=...) -- no trainer.fit call at all).

#SBATCH --job-name=amelia_eval_kbos50_test
#SBATCH --output=eval_kbos_50_test_%j.out
#SBATCH --error=eval_kbos_50_test_%j.err

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

python -m amelia_tf.eval --config-name=eval_kbos
