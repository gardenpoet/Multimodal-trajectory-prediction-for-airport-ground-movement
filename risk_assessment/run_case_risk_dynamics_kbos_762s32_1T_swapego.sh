#!/bin/bash
# Both-sides-predicted risk comparison, ambiguous-intention headline case
# (batch=762, sample=32, real min separation 77.4m), 1T checkpoint. Same
# scene/pair as run_case_risk_dynamics_kbos_762s32_1T.sh, but with ego and
# the interactive agent swapped (ego_agent_id=1, ref back to 0).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_762s32_1T_swapego.sh

#SBATCH --job-name=case_dyn_kbos_762s32_1T_swapego
#SBATCH --output=risk_assessment/case_dyn_kbos_762s32_1T_swapego_%j.out
#SBATCH --error=risk_assessment/case_dyn_kbos_762s32_1T_swapego_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_762_32_dynamics_1T_swapego.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    +data.dataset.config.ego_agent_id=1 \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=762 +case_sample_idx=32 +case_ref_agent_idx=0 \
    +output_json="$OUT_JSON"
