#!/bin/bash
# Continuous-prediction grid (advisor-requested 5x5: 5 models x 5 current-time
# frames, 2s apart, same confirmed near-miss case KBOS b7/s62 / scene
# kbos/KBOS_701_1675098000) -- this is the frame=2178 / 1T cell.
# batch_idx/sample_idx taken from the TP-4T full-test-set CSV's own indexing
# for this frame (same dataloader ordering is reused for 1T/2T/4T, same
# convention as the existing b3s70/b300s35 case scripts).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_b7s62_f2178_1T.sh

#SBATCH --job-name=case_dyn_kbos_b7s62_f2178_1T
#SBATCH --output=risk_assessment/case_dyn_kbos_b7s62_f2178_1T_%j.out
#SBATCH --error=risk_assessment/case_dyn_kbos_b7s62_f2178_1T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b7s62_f2178_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=455 +case_sample_idx=46 \
    +output_json="$OUT_JSON"
