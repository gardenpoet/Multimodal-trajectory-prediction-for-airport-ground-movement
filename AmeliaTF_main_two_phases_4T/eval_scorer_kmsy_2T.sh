#!/bin/bash
# Trains a learned trajectory-selection scorer for KMSY's 2T (num_hypotheses=2,
# 50s-horizon) checkpoint: frozen backbone, only the score-head parameters are
# trained (amelia_tf/eval_two_stage.py's _stage_score, scorer.stage=score),
# then tests with score-based hypothesis selection.
#
# Mirrors eval_scorer_kmsy_4T.sh, but points at the 2T trajectory checkpoint
# trained ~2026-09-06 (predates the 2026-09-17 num_hypotheses config-bug fix,
# but per num_hypotheses_config_bug.md this run intended K=2 anyway, so it is
# unaffected by that bug). The default eval_two_stage.yaml hardcodes
# traj_ckpt_path to a "_4_50" suffix, so it must be explicitly overridden here
# to the 2T checkpoint. mode_ckpt_path is also overridden, to the mode model
# trained most recently (kmsy2_twophases_50.ckpt, per train_two_stage_kmsy.yaml's
# naming, i.e. no hypothesis-count segment) rather than the older
# kmsy2_twophases_4_50.ckpt that configs/eval_two_stage.yaml defaults to;
# mode classification does not depend on num_hypotheses, so the same mode
# checkpoint is shared with eval_scorer_kmsy_4T.sh. scorer.score_head_save is
# overridden so this run's output does not collide with (overwrite) the 4T
# scorer's.
#
# Requires the checkpoints already on disk at:
#   ${ckpt_dir}/Single-Airport/kmsy2/mode_model/kmsy2_twophases_50.ckpt
#   ${ckpt_dir}/Single-Airport/kmsy2/traj_model/kmsy2_twophases_2_50.ckpt
# If the 2T checkpoint is not actually at that path, override traj_ckpt_path
# below with the real one before submitting.

#SBATCH --job-name=amelia_eval_scorer_2T
#SBATCH --output=eval_scorer_kmsy_2T_%j.out
#SBATCH --error=eval_scorer_kmsy_2T_%j.err

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
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_2_50.ckpt' \
    model.traj_net.config.num_hypotheses=2 \
    model.traj_net.config.enable_score_head=true \
    model.traj_net.config.score_mode=5 \
    model.traj_net.config.score_head_type=attention \
    scorer.stage=score \
    scorer.score_head_save='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_soft_2T.pt'
