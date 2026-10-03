#!/bin/bash
# EGO-FIX re-run (dense 50-frame continuous-prediction series, b3/s70 (near-zero-probability alternative)), frame=202 / 2T cell.
# Pins ego to agent_idx=1 (not the dataset own auto-detected
# slot 0) so this frame predicts the SAME physical aircraft tracked across
# this case own window; the interactive/ref agent is left on
# auto-detect (closest approach), since only ego identity must stay
# fixed. See risk_assessment/EGOFIX_NOTES.md for how agent_idx was
# determined (position-continuity matching against the headline frame).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_b3s70_f202_2T_egofix.sh

#SBATCH --job-name=case_dyn_kbos_b3s70_f202_2T_egofix
#SBATCH --output=risk_assessment/case_dyn_kbos_b3s70_f202_2T_egofix_%j.out
#SBATCH --error=risk_assessment/case_dyn_kbos_b3s70_f202_2T_egofix_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b3s70_f202_dynamics_2T_egofix.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    +data.dataset.config.ego_agent_id=1 \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_2_50.ckpt' \
    model.traj_net.config.num_hypotheses=2 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard_2T.pt' \
    +case_batch_idx=491 +case_sample_idx=48 \
    +output_json="$OUT_JSON"
