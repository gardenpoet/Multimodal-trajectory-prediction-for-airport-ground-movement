#!/bin/bash
# Cross-model trajectory dump (AmeliaTF_main plain baseline) for KBOS
# replacement case (batch=762, sample=32) -- see
# run_case_risk_dynamics_kbos_762s32_4T.sh for the full case description.
# Not in any ranked CSV (freshly screened from the full-test-set mc200
# rerun), so scene_file is passed directly via +case_scene_file rather
# than +case_batch_idx/+cases_csv lookup
# (risk_assessment.common.resolve_case_scene_file checks case_scene_file
# first). Scans the FULL test set -- give it ample time. ckpt_path comes
# from configs/eval_kbos.yaml's own default.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_762s32_amelia_baseline.sh

#SBATCH --job-name=case_trajectories_kbos_762s32_baseline
#SBATCH --output=risk_assessment/case_trajectories_kbos_762s32_baseline_%j.out
#SBATCH --error=risk_assessment/case_trajectories_kbos_762s32_baseline_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_762_32_trajectories_amelia_baseline.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kbos \
    +data.dataset.config.random_ego=false \
    +case_scene_file=kbos/KBOS_148_1673060400/001137_n-5.pkl \
    +output_json="$OUT_JSON"
