#!/bin/bash
#SBATCH --job-name=amelia_train
#SBATCH --output=mode_klax_20.out
#SBATCH --error=mode_klax_20.err

#SBATCH --partition=sae
#SBATCH --account=pilot_sae_gpu

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

# Stage 1 only: trains the mode classifier and copies the best checkpoint to
# mode_ckpt_path (configs/train_two_stage_klax_20.yaml), then exits before
# Stage 2. Run this once, before submitting the three train_traj_klax_20.sh
# num_futures sweeps (1T/2T/4T), which reuse this checkpoint.
python -m amelia_tf.train_two_stage_klax_20 skip_traj_training=true
