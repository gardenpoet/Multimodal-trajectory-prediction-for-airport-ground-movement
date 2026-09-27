#!/bin/bash
# Case-study deep dive for KMSY's confirmed candidate (batch=18, sample=27,
# ego_id=0, ref agent auto-detected -- predicted (36.03m) and realized
# (35.58m) separation agree it's a genuine close approach, Aircraft-
# Aircraft, on the movement-area network, mode predicted correctly
# (TurnRight)). Same checkpoint/score-head config as
# run_find_cases_kmsy_50_4T.sh. Re-selected AGAIN 2026-09-27 (see
# run_case_risk_dynamics_kmsy_1T.sh's comment): the (292,46) candidate that
# replaced the original (2708,36) one had a genuine data problem too --
# its realized ego history barely moved at all (0.69m net displacement
# over 10s, noise-scale) -- not just a rendering bug. This one has hist
# displacement=80.0m, straightness=0.95 (see rank_candidates.py), at the
# cost of a looser (but still within the 50m safety margin) separation.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kmsy.sh

#SBATCH --job-name=case_risk_dynamics_kmsy_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kmsy_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kmsy_4T_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_18_27_dynamics_4T.json"
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
    +case_batch_idx=18 +case_sample_idx=27 \
    +output_json="$OUT_JSON"
