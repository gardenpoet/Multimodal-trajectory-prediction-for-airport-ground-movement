#!/bin/bash
# Trains a learned trajectory-selection scorer for KMSY's 20s-horizon 4T
# (num_hypotheses=4) checkpoint: frozen backbone, only the score-head
# parameters are trained (amelia_tf.eval_two_stage's _stage_score,
# scorer.stage=score), then tests with score-based hypothesis selection in
# the same job (12h time limit, no companion _test.sh needed).
#
# Mirrors eval_scorer_kmsy_20_2T.sh, but points at the 4T trajectory
# checkpoint. See that script's header comment for why both mode_ckpt_path
# and traj_ckpt_path must be explicitly overridden for this horizon (the
# base eval_two_stage.yaml's defaults are hardcoded to a "_50" suffix).
#
# enable_score_head/score_mode/score_head_type need the Hydra "+" prefix AND
# must be nested under .decoder (see score_head_config_nesting.md / jobs
# 27758086, 27805912 for the two distinct KMSY 50s failures this fixes).
# num_hypotheses is read at the top level (config.num_hypotheses), so it is
# NOT nested under decoder below.
#
# Requires the checkpoints already on disk at:
#   ${ckpt_dir}/Single-Airport/kmsy_20/mode_model/kmsy_20_twophases_20.ckpt
#   ${ckpt_dir}/Single-Airport/kmsy_20/traj_model/kmsy_20_twophases_4_20.ckpt

#SBATCH --job-name=amelia_eval_scorer_kmsy20_4T
#SBATCH --output=eval_scorer_kmsy_20_4T_%j.out
#SBATCH --error=eval_scorer_kmsy_20_4T_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1
export DEBUG_PHASE_CHECK=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python -m amelia_tf.eval_two_stage \
    ckpt=kmsy_20 \
    data=kmsy2.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_20.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_20.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    scorer.stage=score \
    scorer.score_head_save='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard.pt'
