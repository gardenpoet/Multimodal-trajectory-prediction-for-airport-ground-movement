#!/bin/bash
# Case-study deep dive for KMSY's confirmed candidate (batch=2708, sample=36,
# ego_id=0, ref agent idx=1), 1T checkpoint (1 candidate per mode, 4 total
# hypotheses -- no score head, K=1 so cand_probs is trivially 1.0 per mode).
# Same batch_idx/sample_idx as the 4T run: 1T/2T/4T are different checkpoints
# of the SAME repo/config (AmeliaTF_main_two_phases_4T, data=kmsy.yaml), so
# the scene ordering is identical -- unlike a cross-repo comparison (see
# get_scene_files.py), no scene_file lookup is needed here.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kmsy_1T.sh

#SBATCH --job-name=case_risk_dynamics_kmsy_1T
#SBATCH --output=risk_assessment/case_risk_dynamics_kmsy_1T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kmsy_1T_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_2708_36_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kmsy2 \
    +data.dataset.config.random_ego=false \
    data=kmsy.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=2708 +case_sample_idx=36 +case_ref_agent_idx=1 \
    +output_json="$OUT_JSON"
