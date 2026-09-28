#!/bin/bash
# Case-study deep dive for one of the 2026-09-28 finalized KMSY
# candidates (batch=224, sample=127, sep=44.0m, ego_id/ref
# agent auto-detected -- see case_risk_dynamics.py), 1T checkpoint.
# Hand-picked from the plot_risk_cases.py/plot_risk_cases_pred.py bulk
# screenshot screening.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kmsy_b224s127_1T.sh

#SBATCH --job-name=case_risk_dynamics_kmsy_b224s127_1T
#SBATCH --output=risk_assessment/case_risk_dynamics_kmsy_b224s127_1T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kmsy_b224s127_1T_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_224_127_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kmsy2 \
    +data.dataset.config.random_ego=false \
    data=kmsy.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=224 +case_sample_idx=127 \
    +output_json="$OUT_JSON"
