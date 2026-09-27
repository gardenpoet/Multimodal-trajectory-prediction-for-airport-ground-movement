#!/bin/bash
# Case-study deep dive for KMSY's confirmed candidate (batch=18, sample=27,
# ego_id=0, ref agent auto-detected -- predicted (36.03m) and realized
# (35.58m) separation agree, Aircraft-Aircraft, on the movement-area
# network, mode predicted correctly (TurnRight)). Same batch_idx/sample_idx
# as the 4T run: 1T/2T/4T are different checkpoints of the SAME repo/config
# (AmeliaTF_main_two_phases_4T, data=kmsy.yaml), so the scene ordering is
# identical -- unlike a cross-repo comparison, no scene_file lookup is
# needed here. Re-selected AGAIN 2026-09-27: the (292,46) candidate that
# replaced the original (2708,36) one was itself a poor case-study plot --
# its realized ego history barely moved (0.69m net displacement over 10s,
# noise-scale). This one has hist displacement=80.0m, straightness=0.95 --
# see rank_candidates.py. Its separation (35-36m) is looser than the KBOS/
# KLAX picks (well within the 50m safety margin, but not razor-thin) -- see
# rank_candidates.py's output for why (KMSY's sub-10m candidates were all
# Hold-mode, i.e. near-stationary, the same legibility problem).
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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_18_27_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kmsy2 \
    +data.dataset.config.random_ego=false \
    data=kmsy.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=18 +case_sample_idx=27 \
    +output_json="$OUT_JSON"
