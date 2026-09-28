#!/bin/bash
# Group B illustrative case for KBOS (batch=5, sample=9, ego_id/
# ref agent auto-detected). From the bulk group_b screening
# (risk_assessment.common.select_candidates's criterion="group_b" -- see
# run_plot_group_b_cases_kbos.sh): realized/gt-mode outcome is safe
# (~53.9m) but some OTHER mode/candidate lands at just ~2.4m.
# Demonstrates why risk_worst_case looks beyond the single most-likely
# prediction. 1T checkpoint -- same 1T/2T/4T treatment as the
# confirmed near-miss case, for consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b5s9_1T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b5s9_1T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b5s9_1T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b5s9_1T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_5_9_groupb_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=5 +case_sample_idx=9 \
    +output_json="$OUT_JSON"
