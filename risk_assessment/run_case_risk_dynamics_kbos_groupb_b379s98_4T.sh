#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=379, sample=98,
# ego_id/ref agent auto-detected). Replaces b359/s79 (verified
# 2026-09-29: genuinely ambiguous with a clean naive < gated <
# worst-case story, but its Paper Figures rendering turned out visually
# cluttered -- large, overlapping sigma-spread bands across the three
# feasible modes' candidates) and the earlier b75/s120 pick (genuinely
# ambiguous but essentially risk-free throughout, min predicted
# separation ~131m for every mode/candidate/model). This one comes from
# kbos_50_4T_cases_ranked.csv's own ambiguous=True column (top-2
# feasible-mode mode_prob gap < 0.10): argmax_mode=TurnRight vs
# gt_mode=Straight (a genuine mode_error), realized outcome safe
# (true_min_sep_gt~83.7m) but under the old 50m mean-only metric
# risk_naive~32.2/1000 (~17.8m separation, a real but moderate approach)
# while risk_gated==risk_worst_case~49.2/1000 (~0.8m, near-collision):
# gating fully recovers a much more severe danger that naive alone
# substantially understates, a clean two-tier story. 4T
# checkpoint -- same 1T/2T/4T treatment as the other cases, for
# consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b379s98_4T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b379s98_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b379s98_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b379s98_4T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_379_98_groupb_dynamics_4T.json"
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
    +case_batch_idx=379 +case_sample_idx=98 \
    +output_json="$OUT_JSON"
