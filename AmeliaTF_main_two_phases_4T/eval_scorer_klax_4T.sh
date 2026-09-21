#!/bin/bash
# Trains a learned trajectory-selection scorer for KLAX's 4T (num_hypotheses=4,
# 50s-horizon) checkpoint: frozen backbone, only the score-head parameters are
# trained (amelia_tf/eval_two_stage.py's _stage_score, scorer.stage=score),
# then tests with score-based hypothesis selection in the same job.
#
# Mirrors eval_scorer_kbos_4T.sh/eval_scorer_kmsy_4T.sh. Uses a 12-hour time
# limit so training + test both finish in one job (no companion _test.sh
# script needed), unlike KMSY's original 4-hour scripts.
#
# mode_ckpt_path is overridden to ${ckpt}_twophases_50.ckpt (no hypothesis-
# count segment, since mode classification doesn't depend on num_futures) --
# the base configs/eval_two_stage.yaml default hardcodes mode_ckpt_path to a
# "_4_50" suffix, which is wrong for the mode model. traj_ckpt_path is NOT
# overridden: the base config's default already points at the correct
# "_4_50" trajectory checkpoint for the 4T config.
#
# enable_score_head/score_mode/score_head_type need the Hydra "+" prefix AND
# must be nested under .decoder (see score_head_config_nesting.md / jobs
# 27758086, 27805912 for the two distinct KMSY failures this fixes).
# num_hypotheses is read at the top level (config.num_hypotheses), so it is
# NOT nested under decoder below.
#
# Requires the checkpoints already on disk at:
#   ${ckpt_dir}/Single-Airport/klax2/mode_model/klax2_twophases_50.ckpt
#   ${ckpt_dir}/Single-Airport/klax2/traj_model/klax2_twophases_4_50.ckpt

#SBATCH --job-name=amelia_eval_scorer_klax_4T
#SBATCH --output=eval_scorer_klax_4T_%j.out
#SBATCH --error=eval_scorer_klax_4T_%j.err

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
    ckpt=klax2 \
    data=klax.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    scorer.stage=score \
    scorer.score_head_save='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard.pt'
