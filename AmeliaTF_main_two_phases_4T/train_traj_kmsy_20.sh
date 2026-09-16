#!/bin/bash
#SBATCH --job-name=amelia_train
#SBATCH --output=traj_kmsy_20_%j.out
#SBATCH --error=traj_kmsy_20_%j.err

#SBATCH --partition=sae
#SBATCH --account=pilot_sae_gpu

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=120:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

# NUM_FUTURES (1, 2, or 4) selects the 1T/2T/4T trajectory-candidate variant.
# Requires train_mode_kmsy_20.sh to have completed first (produces the fixed
# mode checkpoint this run reuses via skip_mode_training=true). Submit 3x:
#   sbatch --export=ALL,NUM_FUTURES=1 train_traj_kmsy_20.sh
#   sbatch --export=ALL,NUM_FUTURES=2 train_traj_kmsy_20.sh
#   sbatch --export=ALL,NUM_FUTURES=4 train_traj_kmsy_20.sh
if [ -z "$NUM_FUTURES" ]; then
    echo "ERROR: NUM_FUTURES is not set."
    echo "Submit with, e.g.: sbatch --export=ALL,NUM_FUTURES=1 $0"
    exit 1
fi

python -m amelia_tf.train_two_stage_kmsy_20 \
    skip_mode_training=true \
    model.traj_net.config.decoder.num_futures=${NUM_FUTURES}
