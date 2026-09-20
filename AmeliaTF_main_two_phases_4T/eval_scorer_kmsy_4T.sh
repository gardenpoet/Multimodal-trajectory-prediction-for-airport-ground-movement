#!/bin/bash
# Trains a learned trajectory-selection scorer for KMSY's 4T (num_hypotheses=4,
# 50s-horizon) checkpoint: frozen backbone, only the score-head parameters are
# trained (amelia_tf/eval_two_stage.py's _stage_score, scorer.stage=score),
# then tests with score-based hypothesis selection.
#
# This is a fresh, fully-reproducible rerun of the "kmsy_select_attention4"
# experiment from 2026-07 (score_head_type=attention), whose exact config was
# never saved anywhere recoverable. score_mode and score_head_type below are
# both explicit even though they match the GMM defaults in gmm.py, so this
# script itself is now the record of what was used.
#
# mode_ckpt_path is overridden to the mode model trained most recently
# (kmsy2_twophases_50.ckpt, per train_two_stage_kmsy.yaml's mode_ckpt_path
# naming, i.e. no hypothesis-count segment), rather than the older
# kmsy2_twophases_4_50.ckpt that configs/eval_two_stage.yaml defaults to.
#
# Requires the checkpoints already on disk at:
#   ${ckpt_dir}/Single-Airport/kmsy2/mode_model/kmsy2_twophases_50.ckpt
#   ${ckpt_dir}/Single-Airport/kmsy2/traj_model/kmsy2_twophases_4_50.ckpt
# (ckpt=kmsy2, matching configs/train_two_stage_kmsy.yaml's traj_ckpt_path)

#SBATCH --job-name=amelia_eval_scorer
#SBATCH --output=eval_scorer_kmsy_4T_%j.out
#SBATCH --error=eval_scorer_kmsy_4T_%j.err

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
export DEBUG_PHASE_CHECK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m amelia_tf.eval_two_stage \
    ckpt=kmsy2 \
    data=kmsy.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    model.traj_net.config.enable_score_head=true \
    model.traj_net.config.score_mode=5 \
    model.traj_net.config.score_head_type=attention \
    scorer.stage=score
