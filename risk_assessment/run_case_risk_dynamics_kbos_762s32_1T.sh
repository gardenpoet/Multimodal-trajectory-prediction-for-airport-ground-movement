#!/bin/bash
# Replacement illustrative case for KBOS (batch=762, sample=32, ego_id/
# ref agent auto-detected), screened from the full-test-set KBOS TP-4T
# mc200 rerun (gated_diverges=True, sorted by risk_gated-risk_naive gap).
# Genuine mode error (gt=TurnRight, argmax=Straight), ambiguous_group=
# {Straight,TurnRight}, naive=0.0 but gated=1.0 -- unlike the old
# b300/s35 case, this one's gated value survives the 2026-09-30
# chain-restricted ambiguous-group redefinition by construction (the
# dangerous candidate sits inside the ambiguous pair itself, not outside
# it). Realized separation ~5.99m (scene_min_sep_gt), tighter than the
# confirmed near-miss case's 7.85m. Replaces the now-invalid b300/s35
# case; see risk_assessment_experiment.md memory. 1T checkpoint -- same
# 1T/2T/4T treatment as the other illustrative cases.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_762s32_1T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_762s32_1T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_762s32_1T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_762s32_1T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_762_32_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=762 +case_sample_idx=32 \
    +output_json="$OUT_JSON"
