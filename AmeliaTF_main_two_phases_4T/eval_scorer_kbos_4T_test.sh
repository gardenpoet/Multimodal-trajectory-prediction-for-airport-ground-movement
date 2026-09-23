#!/bin/bash
# Test-only rerun of the KBOS 4T scorer, to pick up the corrected NLL
# computation (commit 9cee61b: NLL now mixes over the full mode x
# candidate distribution using the scorer's own softmaxed score, instead
# of the previously hard-selected single candidate per mode). Since the
# score head is already trained and saved by eval_scorer_kbos_4T.sh, this
# reloads it via amelia_tf.eval_two_stage's "score_test" stage
# (_stage_score_test) and runs test only -- no need to redo training.
# RMSE/MADE/MFDE/PADE/PFDE are unaffected by that fix and don't need
# rerunning; this is purely to refresh NLL.
#
# Needs the same decoder overrides as eval_scorer_kbos_4T.sh (enable_score_head/
# score_mode/score_head_type nested under .decoder, per score_head_config_nesting.md).
# scorer.score_head_load needs the Hydra "+" prefix (see
# score_head_config_nesting.md's "related, separate bug" note) since it
# isn't declared in configs/eval_two_stage.yaml (only score_head_save is).
#
# Requires the checkpoints already on disk at:
#   ${ckpt_dir}/Single-Airport/kbos2/mode_model/kbos2_twophases_50.ckpt
#   ${ckpt_dir}/Single-Airport/kbos2/traj_model/kbos2_twophases_4_50.ckpt
# and the saved scorer head at:
#   /gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/kbos2/per_mode_scorer_hard.pt

#SBATCH --job-name=amelia_eval_scorer_kbos_4T_test
#SBATCH --output=eval_scorer_kbos_4T_test_%j.out
#SBATCH --error=eval_scorer_kbos_4T_test_%j.err

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
    ckpt=kbos2 \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    scorer.stage=score_test \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard.pt'
