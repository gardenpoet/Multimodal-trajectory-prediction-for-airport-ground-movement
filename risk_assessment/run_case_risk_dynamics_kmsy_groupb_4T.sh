#!/bin/bash
# Group B illustrative case for KMSY (batch=193, sample=127, ego_id=0, ref
# agent auto-detected): gt-mode (TurnRight) prediction (126.2m) reasonably
# matches the realized outcome (103.4m), both safe -- but the Straight
# mode's candidate predicts just 0.02m, i.e. "if this aircraft had gone
# straight instead of turning right, it would have nearly collided."
# Demonstrates why risk_worst_case looks beyond the single most-likely
# prediction. 4T checkpoint only.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kmsy_groupb_4T.sh

#SBATCH --job-name=case_risk_dynamics_kmsy_groupb_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kmsy_groupb_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kmsy_groupb_4T_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_193_127_groupb_dynamics_4T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kmsy2 \
    +data.dataset.config.random_ego=false \
    data=kmsy.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard.pt' \
    +case_batch_idx=193 +case_sample_idx=127 \
    +output_json="$OUT_JSON"
