#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=75, sample=120,
# ego_id/ref agent auto-detected). Unlike the other Group B candidates
# (all screened via criterion="group_b", which requires gt_mode ==
# argmax_mode and so structurally excludes ambiguous cases), this one
# comes straight from the ranked CSV's own ambiguous=True column
# (top-2 feasible-mode mode_prob gap < 0.10): argmax_mode=TurnLeft vs
# gt_mode=Straight (a genuine mode_error), risk_naive=0 but
# risk_gated~0.042 under the old 50m mean-only metric -- demonstrates
# the ambiguity-gated strategy actually diverging from naive, which
# none of the near_miss/group_b picks do. 1T checkpoint -- same
# 1T/2T/4T treatment as the other cases, for consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b75s120_1T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b75s120_1T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b75s120_1T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b75s120_1T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_75_120_groupb_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=75 +case_sample_idx=120 \
    +output_json="$OUT_JSON"
