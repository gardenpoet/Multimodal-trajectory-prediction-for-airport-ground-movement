#!/bin/bash
# Fills the one genuine gap (frame 2175) in the b7/s62 dense series.
# Amelia-TF baseline cell. scene_file n-suffix confirmed to match the
# two-stage model's own naming for this frame (n-5).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_b7s62_f2175_amelia_baseline.sh

#SBATCH --job-name=case_traj_kbos_b7s62_f2175_baseline
#SBATCH --output=risk_assessment/case_traj_kbos_b7s62_f2175_baseline_%j.out
#SBATCH --error=risk_assessment/case_traj_kbos_b7s62_f2175_baseline_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b7s62_f2175_trajectories_amelia_baseline.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kbos \
    +data.dataset.config.random_ego=false \
    +case_scene_file=kbos/KBOS_701_1675098000/000175_n-5.pkl \
    +output_json="$OUT_JSON"
