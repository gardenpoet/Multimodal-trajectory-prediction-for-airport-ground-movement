#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=201, sample=103,
# ego_id/ref agent auto-detected). One of three candidates (with
# b389/s77 and b192/s13) selected via a CSV-only proxy for genuine
# multi-mode probability spread: the fraction of the naive-to-worst-case
# gap that probability-weighted risk is pulled away from naive.
# argmax_mode=Straight vs gt_mode=TurnRight (a genuine mode_error); like
# b192/s13, the REALIZED trajectory itself already came within
# true_min_sep_gt~3.1m of the reference aircraft, while risk_worst_case
# reaches ~43.5/1000 (~6.5m separation) under the old 50m mean-only
# metric (naive=0, i.e. ~50m) and risk_prob_weighted~25.0/1000 (~25.0m
# separation) sits 57.5% of the way from naive toward worst-case,
# evidence of real multi-mode probability spread. 4T checkpoint --
# same 1T/2T/4T treatment as the other cases, for consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b201s103_4T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b201s103_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b201s103_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b201s103_4T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_201_103_groupb_dynamics_4T.json"
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
    +case_batch_idx=201 +case_sample_idx=103 \
    +output_json="$OUT_JSON"
