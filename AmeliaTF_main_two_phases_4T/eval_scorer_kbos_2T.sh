#!/bin/bash
# Trains a learned trajectory-selection scorer for KBOS's 2T (num_hypotheses=2,
# 50s-horizon) checkpoint: frozen backbone, only the score-head parameters are
# trained (amelia_tf/eval_two_stage.py's _stage_score, scorer.stage=score),
# then tests with score-based hypothesis selection in the same job.
#
# Mirrors eval_scorer_kbos_4T.sh, but points at the 2T trajectory checkpoint.
# The base eval_two_stage.yaml hardcodes traj_ckpt_path to a "_4_50" suffix,
# so it must be explicitly overridden here to the 2T checkpoint.
# mode_ckpt_path is overridden the same way as eval_scorer_kbos_4T.sh (mode
# classification does not depend on num_hypotheses, so the same mode
# checkpoint is shared with the 4T script). scorer.score_head_save is
# overridden so this run's output does not collide with (overwrite) the 4T
# scorer's.
#
# Uses a 12-hour time limit so training + test both finish in one job (no
# companion _test.sh script needed), unlike KMSY's original 4-hour scripts.
#
# enable_score_head/score_mode/score_head_type need the Hydra "+" prefix AND
# must be nested under .decoder (see score_head_config_nesting.md / jobs
# 27758086, 27805912 for the two distinct KMSY failures this fixes).
# num_hypotheses is read at the top level (config.num_hypotheses), so it is
# NOT nested under decoder below.
#
# Requires the checkpoints already on disk at:
#   ${ckpt_dir}/Single-Airport/kbos2/mode_model/kbos2_twophases_50.ckpt
#   ${ckpt_dir}/Single-Airport/kbos2/traj_model/kbos2_twophases_2_50.ckpt
# If the 2T checkpoint is not actually at that path, override traj_ckpt_path
# below with the real one before submitting.

#SBATCH --job-name=amelia_eval_scorer_kbos_2T
#SBATCH --output=eval_scorer_kbos_2T_%j.out
#SBATCH --error=eval_scorer_kbos_2T_%j.err

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
    ckpt=kbos2 \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_2_50.ckpt' \
    model.traj_net.config.num_hypotheses=2 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    scorer.stage=score \
    scorer.score_head_save='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard_2T.pt'
