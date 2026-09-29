#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=389, sample=77,
# ego_id/ref agent auto-detected). One of three candidates (with
# b192/s13 and b201/s103) tried after several earlier ambiguous
# mode-error picks all visually reduced to just two clearly-visible
# candidate lines (their third feasible mode's probability was
# negligible). Selected via a new CSV-only proxy: the fraction of the
# naive-to-worst-case gap that probability-weighted risk is pulled away
# from naive (risk_prob_weighted vs risk_naive/risk_worst_case in
# kbos_50_4T_cases_ranked.csv) -- a large pull fraction is only possible
# if real (non-negligible) probability mass sits on a mode other than
# the naive pick, unlike the earlier picks. This one has the largest
# pull fraction found (66.7%): argmax_mode=TurnRight vs gt_mode=Straight
# (a genuine mode_error), realized outcome safe (true_min_sep_gt~76.4m)
# but under the old 50m mean-only metric risk_naive=0 (>=50m separation)
# while risk_worst_case~48.8/1000 (~1.2m, near-collision) and
# risk_prob_weighted~32.6/1000 (~17.4m) sits substantially closer to
# worst-case than to naive, evidence of real multi-mode probability
# spread. 4T checkpoint -- same 1T/2T/4T treatment as the other cases,
# for consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b389s77_4T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b389s77_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b389s77_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b389s77_4T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_389_77_groupb_dynamics_4T.json"
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
    +case_batch_idx=389 +case_sample_idx=77 \
    +output_json="$OUT_JSON"
