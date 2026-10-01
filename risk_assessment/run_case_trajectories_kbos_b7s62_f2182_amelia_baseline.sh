#!/bin/bash
# Dense 50-frame continuous-prediction series, frame=2182 / Amelia-TF
# baseline cell. scene_file passed directly (not in any ranked CSV);
# n-suffix matches this frame's own baseline-pipeline scene chunking.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_b7s62_f2182_amelia_baseline.sh

#SBATCH --job-name=case_traj_kbos_b7s62_f2182_baseline
#SBATCH --output=risk_assessment/case_traj_kbos_b7s62_f2182_baseline_%j.out
#SBATCH --error=risk_assessment/case_traj_kbos_b7s62_f2182_baseline_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b7s62_f2182_trajectories_amelia_baseline.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kbos \
    +data.dataset.config.random_ego=false \
    +case_scene_file=kbos/KBOS_701_1675098000/002182_n-5.pkl \
    +output_json="$OUT_JSON"
