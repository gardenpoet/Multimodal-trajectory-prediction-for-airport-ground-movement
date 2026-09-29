#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=192, sample=13,
# ego_id/ref agent auto-detected). One of three candidates (with
# b389/s77 and b201/s103) selected via a CSV-only proxy for genuine
# multi-mode probability spread: the fraction of the naive-to-worst-case
# gap that probability-weighted risk is pulled away from naive (large
# only if real probability mass sits on a mode other than the naive
# pick). The most compelling of the three: argmax_mode=Straight vs
# gt_mode=TurnRight (a genuine mode_error), and unlike every other case
# tried so far, the REALIZED trajectory itself already came within
# true_min_sep_gt~2.2m of the reference aircraft -- closer than the
# confirmed-near-miss case's own 7.85m -- while risk_worst_case reaches
# ~47.8/1000 (~2.2m separation, essentially matching how close the
# realised outcome itself came) under the old 50m mean-only metric
# (naive=0, i.e. the as-deployed pick showed no danger at all) and
# risk_prob_weighted ~30.2/1000 (~19.8m separation) is pulled 63% of
# the way from naive toward
# worst-case, evidence of real multi-mode probability spread rather
# than one dominant mode plus negligible alternatives. 4T checkpoint --
# same 1T/2T/4T treatment as the other cases, for consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b192s13_4T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b192s13_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b192s13_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b192s13_4T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_192_13_groupb_dynamics_4T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard.pt' \
    +case_batch_idx=192 +case_sample_idx=13 \
    +output_json="$OUT_JSON"
