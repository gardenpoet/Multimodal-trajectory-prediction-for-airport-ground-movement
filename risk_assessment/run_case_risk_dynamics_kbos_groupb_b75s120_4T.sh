#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=75, sample=120,
# ego_id/ref agent auto-detected). Unlike the other Group B candidates
# (all screened via criterion="group_b", which requires gt_mode ==
# argmax_mode and so structurally excludes ambiguous cases -- verified
# 2026-09-29 that none of the confirmed/near_miss/group_b picks ever
# have ambiguous=True), this one comes straight from
# kbos_50_4T_cases_ranked.csv's own ambiguous=True column (top-2
# feasible-mode mode_prob gap < 0.10): argmax_mode=TurnLeft vs
# gt_mode=Straight (a genuine mode_error), risk_naive=0 but
# risk_gated~0.042 and risk_worst_case~0.048 under the old 50m
# mean-only metric -- demonstrates the ambiguity-gated strategy
# actually diverging from naive (and sitting between naive and
# worst-case), which none of the near_miss/group_b picks do. 4T
# checkpoint -- same 1T/2T/4T treatment as the other cases, for
# consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b75s120_4T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b75s120_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b75s120_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b75s120_4T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_75_120_groupb_dynamics_4T.json"
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
    +case_batch_idx=75 +case_sample_idx=120 \
    +output_json="$OUT_JSON"
