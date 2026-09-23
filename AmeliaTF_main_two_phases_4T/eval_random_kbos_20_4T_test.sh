#!/bin/bash
# Random-selection baseline for KBOS H20 4T. See eval_random_kmsy_2T_test.sh
# for the full rationale.

#SBATCH --job-name=amelia_eval_random_kbos20_4T_test
#SBATCH --output=eval_random_kbos_20_4T_test_%j.out
#SBATCH --error=eval_random_kbos_20_4T_test_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m amelia_tf.eval_two_stage \
    ckpt=kbos_20 \
    data=kbos2.yaml \
    paths=default2.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_20.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_20.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.pred_len=20 \
    scorer.stage=eval \
    +model.extra_params.selection_mode=random
