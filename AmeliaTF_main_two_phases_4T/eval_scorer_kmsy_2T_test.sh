#!/bin/bash
# Test-only rerun of the KMSY 2T scorer: eval_scorer_kmsy_2T.sh (job
# 27806973) completed scorer training and saved per_mode_scorer_hard_2T.pt,
# but the final test phase was killed by the 4-hour time limit before
# finishing. Rather than redo training, use amelia_tf.eval_two_stage's
# "score_test" stage (_stage_score_test), which reloads the pristine 2T
# backbone plus the already-saved score head and runs test only.
#
# Needs the same decoder overrides as eval_scorer_kmsy_2T.sh (enable_score_head/
# score_mode/score_head_type nested under .decoder, per score_head_config_nesting.md)
# so the GMM is rebuilt with the score-head submodules present -- otherwise
# load_state_dict on the saved blob has nothing matching to load into.
#
# Requires the checkpoints already on disk at:
#   ${ckpt_dir}/Single-Airport/kmsy2/mode_model/kmsy2_twophases_50.ckpt
#   ${ckpt_dir}/Single-Airport/kmsy2/traj_model/kmsy2_twophases_2_50.ckpt
# and the saved scorer head at:
#   /gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/kmsy2/per_mode_scorer_hard_2T.pt
#
# scorer.score_head_load needs the Hydra "+" prefix (job 27832482 failed on
# this for the 4T variant of this script): configs/eval_two_stage.yaml only
# declares scorer.score_head_save, never score_head_load, even though
# eval_two_stage.py's cfg.scorer.get("score_head_load", ...) call handles
# it being absent just fine.

#SBATCH --job-name=amelia_eval_scorer_2T_test
#SBATCH --output=eval_scorer_kmsy_2T_test_%j.out
#SBATCH --error=eval_scorer_kmsy_2T_test_%j.err

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
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    scorer.stage=score_test \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard_2T.pt'
