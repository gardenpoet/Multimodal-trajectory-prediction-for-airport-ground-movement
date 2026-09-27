#!/bin/bash
# Case-study deep dive for KLAX's confirmed candidate (batch=65, sample=41,
# ego_id=0, ref agent auto-detected -- predicted (4.23m) and realized (9.10m)
# separation agree it's a genuine close approach, Aircraft-Aircraft, on the
# movement-area network, mode predicted correctly (Straight)). Re-selected
# 2026-09-27: the original (317,74) candidate was a genuine model-failure
# illustration (predicted 47.25m vs realized 2.48m) but its trajectory map
# was never actually inspected before that call -- this one instead
# prioritises a legible plot (straightness=0.99, hist displacement=50.1m --
# see rank_candidates.py); the "model underestimates a real near-miss"
# story, if still wanted, needs a separate candidate. 1T checkpoint (1
# candidate per mode, 4 total hypotheses).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_klax_1T.sh

#SBATCH --job-name=case_risk_dynamics_klax_1T
#SBATCH --output=risk_assessment/case_risk_dynamics_klax_1T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_klax_1T_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/klax_case_65_41_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=klax2 \
    +data.dataset.config.random_ego=false \
    data=klax.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=65 +case_sample_idx=41 \
    +output_json="$OUT_JSON"
