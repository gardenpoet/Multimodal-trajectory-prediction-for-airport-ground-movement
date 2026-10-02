#!/bin/bash
# Both-sides-predicted risk comparison, confirmed near-miss headline case
# (batch=7, sample=62, scene kbos/KBOS_701_1675098000/002194_n-5.pkl, real
# min separation 22.3m). Same scene/pair as
# run_case_risk_dynamics_kbos_b7s62_4T.sh, but with ego and the
# interactive agent swapped (ego_agent_id=1, ref back to 0) -- requires
# the ego_agent_id fix in AmeliaTF_main_two_phases_4T's amelia_dataset.py
# (random_ego=false previously ignored ego_agent_id and always used
# agent 0).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_b7s62_4T_swapego.sh

#SBATCH --job-name=case_dyn_kbos_b7s62_4T_swapego
#SBATCH --output=risk_assessment/case_dyn_kbos_b7s62_4T_swapego_%j.out
#SBATCH --error=risk_assessment/case_dyn_kbos_b7s62_4T_swapego_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_7_62_dynamics_4T_swapego.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    +data.dataset.config.ego_agent_id=1 \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard.pt' \
    +case_batch_idx=7 +case_sample_idx=62 +case_ref_agent_idx=0 \
    +output_json="$OUT_JSON"
