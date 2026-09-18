#!/bin/bash
#SBATCH --job-name=stgcnn_eval
#SBATCH --output=stgcnn_klax_20_eval.out
#SBATCH --error=stgcnn_klax_20_eval.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

# Eval-only rerun to pick up RMSE + per-mode metrics added after this run's
# original training completed. Checkpoint path from this run's own original
# log: grep "Best ckpt path:" stgcnn_klax_20.out
python -m amelia_tf.train_stgcnn_klax_20 train=false \
    'ckpt_path="/gpfs/scratch/exy064/ljx/Risk-Assessment/STGCNN_baseline/out/logs/train/runs/2026-09-17_19-34-03/checkpoints/epoch_092.ckpt"'
