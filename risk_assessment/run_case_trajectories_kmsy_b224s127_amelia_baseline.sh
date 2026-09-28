#!/bin/bash
# Cross-model trajectory dump (AmeliaTF_main plain baseline) for one of the
# 2026-09-28 finalized KMSY candidates (batch=224, sample=127
# in the two-stage model's own dataloader ordering, sep=44.0m) --
# scene_file is resolved from the ranked CSV at run time (see
# risk_assessment/common.py's resolve_case_scene_file), not hardcoded.
# Scans the FULL test set -- give it ample time. ckpt_path comes from
# configs/eval_kmsy.yaml's own default.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kmsy_b224s127_amelia_baseline.sh

#SBATCH --job-name=case_trajectories_kmsy_b224s127_baseline
#SBATCH --output=risk_assessment/case_trajectories_kmsy_b224s127_baseline_%j.out
#SBATCH --error=risk_assessment/case_trajectories_kmsy_b224s127_baseline_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env
module load cuda/12.2.2-gcc-12.2.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_case_224_127_trajectories_amelia_baseline.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kmsy \
    +data.dataset.config.random_ego=false \
    +case_batch_idx=224 +case_sample_idx=127 \
    +cases_csv="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kmsy_50_4T_cases_ranked.csv" \
    +output_json="$OUT_JSON"
