#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=359, sample=79,
# ego_id/ref agent auto-detected). Replaces the earlier b75/s120 pick
# (verified 2026-09-29 to be genuinely ambiguous but essentially
# risk-free throughout -- min predicted separation ~131m for every
# mode/candidate/model, since the reference aircraft is already that
# far away at t=0 and taxi speeds can't close the gap in 50s -- so
# naive/worst-case/prob-weighted/gated all collapse to ~0 and it
# demonstrates nothing about risk aggregation). This one comes from
# kbos_50_4T_cases_ranked.csv's own ambiguous=True column (top-2
# feasible-mode mode_prob gap < 0.10): argmax_mode=Straight vs
# gt_mode=TurnRight (a genuine mode_error), and under the old 50m
# mean-only metric risk_naive~46.2/1000 (~3.8m separation) already
# shows real concern, risk_gated~49.4/1000 (~0.6m) catches the
# feasible-mode top-picks' worse candidate, and risk_worst_case~49.9/
# 1000 (~0.1m, near-collision) digs into an even lower-probability
# candidate that gating alone doesn't reach -- a clean three-tier
# naive < gated < worst-case story that b75/s120 couldn't provide. 4T
# checkpoint -- same 1T/2T/4T treatment as the other cases, for
# consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b359s79_4T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b359s79_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b359s79_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b359s79_4T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_359_79_groupb_dynamics_4T.json"
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
    +case_batch_idx=359 +case_sample_idx=79 \
    +output_json="$OUT_JSON"
