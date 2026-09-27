#!/bin/bash
# Cross-model trajectory dump for KBOS's confirmed candidate
# (scene_file=kbos/KBOS_148_1673060400/002462_n-7.pkl), AmeliaTF_main
# (plain, non-manoeuvre-conditioned) baseline. Scans the FULL test set --
# give it ample time. ckpt_path comes from configs/eval_kbos.yaml's own
# default (same as run_find_cases_kbos_50_amelia_baseline.sh).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_amelia_baseline.sh

#SBATCH --job-name=case_trajectories_kbos_baseline
#SBATCH --output=risk_assessment/case_trajectories_kbos_baseline_%j.out
#SBATCH --error=risk_assessment/case_trajectories_kbos_baseline_%j.err

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

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_103_47_trajectories_amelia_baseline.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kbos \
    +data.dataset.config.random_ego=false \
    +case_scene_file=kbos/KBOS_148_1673060400/002462_n-7.pkl \
    +output_json="$OUT_JSON"
