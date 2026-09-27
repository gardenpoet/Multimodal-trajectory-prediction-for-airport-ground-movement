#!/bin/bash
# Case-study deep dive for KBOS's confirmed candidate (batch=103, sample=47,
# ego_id=0, ref agent auto-detected -- predicted (7.94m) and realized (2.12m)
# separation agree it's a genuine close approach, Aircraft-Aircraft, on the
# movement-area network; ALSO mode_error=True & ambiguous=True, so this case
# doubles as the "gating matters" illustration). 1T checkpoint (1 candidate
# per mode, 4 total hypotheses -- no score head).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_1T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_1T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_1T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_1T_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_103_47_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=103 +case_sample_idx=47 \
    +output_json="$OUT_JSON"
