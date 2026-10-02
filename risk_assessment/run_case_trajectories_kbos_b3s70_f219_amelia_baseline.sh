#!/bin/bash
# Dense 50-frame continuous-prediction series (b3/s70), frame=219 / Amelia-TF baseline cell.
# scene_file's n-suffix looked up per frame from the bulk CSV (NOT
# hardcoded n-5 -- that was a real bug in the first generation of this
# script, which caused every frame in this case to fail, since this
# scenario's real agent count is never 5).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_b3s70_f219_amelia_baseline.sh

#SBATCH --job-name=case_traj_kbos_b3s70_f219_baseline
#SBATCH --output=risk_assessment/case_traj_kbos_b3s70_f219_baseline_%j.out
#SBATCH --error=risk_assessment/case_traj_kbos_b3s70_f219_baseline_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b3s70_f219_trajectories_amelia_baseline.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kbos \
    +data.dataset.config.random_ego=false \
    +case_scene_file=kbos/KBOS_183_1673186400/000219_n-10.pkl \
    +output_json="$OUT_JSON"
