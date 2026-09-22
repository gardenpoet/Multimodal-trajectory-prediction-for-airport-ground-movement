#!/bin/bash
# Trains a learned trajectory-selection scorer for KLAX's 20s-horizon 2T
# (num_hypotheses=2) checkpoint: frozen backbone, only the score-head
# parameters are trained (amelia_tf.eval_two_stage's _stage_score,
# scorer.stage=score), then tests with score-based hypothesis selection in
# the same job (12h time limit, no companion _test.sh needed).
#
# 20s-horizon checkpoints live under a SEPARATE ckpt namespace (klax_20, not
# klax2) with its own data config: data=klax2.yaml pulls in configs/data/
# default2.yaml (pred_lens=[10,20], traj_len=30), vs the 50s scripts' plain
# klax.yaml (default.yaml, pred_lens=[20,50]). The base configs/eval_two_
# stage.yaml hardcodes both mode_ckpt_path and traj_ckpt_path to a "_50"
# suffix regardless of horizon, so BOTH must be explicitly overridden here
# (unlike the 50s 4T script, which could rely on the base default by
# coincidence) -- there is no default that is correct for this horizon.
#
# paths=default2.yaml is ALSO required (job 27892446 failed without it on
# the KMSY variant of this script): the base eval_two_stage.yaml defaults
# to paths=default.yaml, whose scenes_dir points at the 50s preprocessed
# scenes (proc_full_scenes/, 60-length windows). data=klax2.yaml alone
# correctly sets traj_len=30 in the model config (so encoder_config.
# T_size=30), but without also overriding paths, the datamodule still
# loads the wrong (60-length) scene files from disk, so the model errors
# with "Sequence length 60 exceeds maximum block size 30" -- paths=
# default2.yaml points scenes_dir at proc_full_scenes2/, the actual
# 20s-horizon preprocessed data.
#
# +model.traj_net.config.decoder.pred_len=20 is ALSO required (job
# 27959063 failed without it, on this exact KLAX 2T script): gmm.py's
# attention score head reads config.decoder.pred_len to size its "full"
# key encoder at init time (getattr(config, "pred_len", getattr(config,
# "T_pred", 50))) -- not declared under configs/model/combined_traj_
# pred.yaml's decoder: block, so it silently defaults to 50
# (coincidentally correct for the 50s scripts) unless explicitly
# overridden here.
#
# enable_score_head/score_mode/score_head_type need the Hydra "+" prefix AND
# must be nested under .decoder (see score_head_config_nesting.md / jobs
# 27758086, 27805912 for the two distinct KMSY 50s failures this fixes).
# num_hypotheses is read at the top level (config.num_hypotheses), so it is
# NOT nested under decoder below.
#
# Requires the checkpoints already on disk at:
#   ${ckpt_dir}/Single-Airport/klax_20/mode_model/klax_20_twophases_20.ckpt
#   ${ckpt_dir}/Single-Airport/klax_20/traj_model/klax_20_twophases_2_20.ckpt

#SBATCH --job-name=amelia_eval_scorer_klax20_2T
#SBATCH --output=eval_scorer_klax_20_2T_%j.out
#SBATCH --error=eval_scorer_klax_20_2T_%j.err

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
    ckpt=klax_20 \
    data=klax2.yaml \
    paths=default2.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_20.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_2_20.ckpt' \
    model.traj_net.config.num_hypotheses=2 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +model.traj_net.config.decoder.pred_len=20 \
    scorer.stage=score \
    scorer.score_head_save='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard_2T.pt'
