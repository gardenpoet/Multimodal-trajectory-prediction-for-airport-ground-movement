#!/bin/bash
# Case-study deep dive for KMSY's confirmed candidate (batch=2708, sample=36,
# ego_id=0, ref agent idx=1), 2T checkpoint (2 candidates per mode, 8 total
# hypotheses). Same batch_idx/sample_idx as the 4T/1T runs -- see
# run_case_risk_dynamics_kmsy_1T.sh's comment for why no scene_file lookup
# is needed here (same repo/config, just a different checkpoint).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kmsy_2T.sh

#SBATCH --job-name=case_risk_dynamics_kmsy_2T
#SBATCH --output=risk_assessment/case_risk_dynamics_kmsy_2T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kmsy_2T_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_2708_36_dynamics_2T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kmsy2 \
    +data.dataset.config.random_ego=false \
    data=kmsy.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_2_50.ckpt' \
    model.traj_net.config.num_hypotheses=2 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard_2T.pt' \
    +case_batch_idx=2708 +case_sample_idx=36 +case_ref_agent_idx=1 \
    +output_json="$OUT_JSON"
