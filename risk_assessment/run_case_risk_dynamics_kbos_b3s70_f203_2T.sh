#!/bin/bash
# Dense 50-frame continuous-prediction series (b3/s70 (near-zero-probability alternative)), frame=203 / 2T cell.
# Part of the 184-233 gap-free extension for this case, same pattern as
# the b7/s62 series (run_case_risk_dynamics_kbos_b7s62_f2176_1T.sh).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_b3s70_f203_2T.sh

#SBATCH --job-name=case_dyn_kbos_b3s70_f203_2T
#SBATCH --output=risk_assessment/case_dyn_kbos_b3s70_f203_2T_%j.out
#SBATCH --error=risk_assessment/case_dyn_kbos_b3s70_f203_2T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b3s70_f203_dynamics_2T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_2_50.ckpt' \
    model.traj_net.config.num_hypotheses=2 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard_2T.pt' \
    +case_batch_idx=634 +case_sample_idx=65 \
    +output_json="$OUT_JSON"
