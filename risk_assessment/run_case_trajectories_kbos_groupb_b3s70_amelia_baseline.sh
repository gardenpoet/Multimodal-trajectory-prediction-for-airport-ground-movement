#!/bin/bash
# Cross-model trajectory dump (AmeliaTF_main plain baseline) for KBOS
# group_b candidate (batch=3, sample=70 in the two-stage
# model's own dataloader ordering) -- see
# run_case_risk_dynamics_kbos_groupb_b3s70_4T.sh. scene_file is resolved
# from the ranked CSV at run time
# (risk_assessment.common.resolve_case_scene_file), not hardcoded. Scans
# the FULL test set -- give it ample time. ckpt_path comes from
# configs/eval_kbos.yaml's own default.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_groupb_b3s70_amelia_baseline.sh

#SBATCH --job-name=case_trajectories_kbos_groupb_b3s70_baseline
#SBATCH --output=risk_assessment/case_trajectories_kbos_groupb_b3s70_baseline_%j.out
#SBATCH --error=risk_assessment/case_trajectories_kbos_groupb_b3s70_baseline_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_3_70_groupb_trajectories_amelia_baseline.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kbos \
    +data.dataset.config.random_ego=false \
    +case_batch_idx=3 +case_sample_idx=70 \
    +cases_csv="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_4T_cases_ranked.csv" \
    +output_json="$OUT_JSON"
