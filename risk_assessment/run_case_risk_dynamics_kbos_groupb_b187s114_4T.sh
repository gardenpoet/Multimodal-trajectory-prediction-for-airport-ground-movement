#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=187, sample=114,
# ego_id/ref agent auto-detected). Evaluated alongside b379/s98 (which
# rendered cleanly, verified 2026-09-29) and b373/s25 as alternative
# picks with a REALIZED near-miss rather than a purely hypothetical
# candidate-pool danger, comes from kbos_50_4T_cases_ranked.csv's own
# ambiguous=True column (top-2 feasible-mode mode_prob gap < 0.10):
# argmax_mode=TurnRight vs gt_mode=Straight (a genuine mode_error).
# Under the old 50m mean-only metric, the realized trajectory itself
# already came within true_min_sep_gt~18.2m of the reference aircraft,
# risk_naive~30.6/1000 (~19.4m separation) understates this further,
# and risk_gated~44.6/1000 (~5.4m) sits close to risk_worst_case~44.7/
# 1000 (~5.3m) -- gating recovers nearly all of the worst-case danger
# in a scenario that was already a genuine close call, not just a
# hypothetical one. 4T checkpoint -- same 1T/2T/4T treatment as the
# other cases, for consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b187s114_4T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b187s114_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b187s114_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b187s114_4T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_187_114_groupb_dynamics_4T.json"
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
    +case_batch_idx=187 +case_sample_idx=114 \
    +output_json="$OUT_JSON"
