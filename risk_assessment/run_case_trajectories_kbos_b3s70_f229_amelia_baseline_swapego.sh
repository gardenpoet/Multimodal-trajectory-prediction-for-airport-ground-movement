#!/bin/bash
# Both-sides-predicted risk comparison, b3/s70 headline case (frame 229):
# same scene/pair as run_case_trajectories_kbos_b3s70_f229_amelia_baseline.sh,
# but with ego and the interactive agent swapped (ego_agent_id=1, ref back
# to 0) -- requires the ego_agent_id fix in AmeliaTF_main's
# amelia_dataset.py (random_ego=false previously ignored ego_agent_id and
# always used agent 0).
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_trajectories_kbos_b3s70_f229_amelia_baseline_swapego.sh

#SBATCH --job-name=case_traj_kbos_b3s70_f229_baseline_swapego
#SBATCH --output=risk_assessment/case_traj_kbos_b3s70_f229_baseline_swapego_%j.out
#SBATCH --error=risk_assessment/case_traj_kbos_b3s70_f229_baseline_swapego_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_b3s70_f229_trajectories_amelia_baseline_swapego.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_trajectories_amelia_baseline \
    --config-name=eval_kbos \
    +data.dataset.config.random_ego=false \
    +data.dataset.config.ego_agent_id=1 \
    +case_scene_file=kbos/KBOS_183_1673186400/000229_n-11.pkl \
    +case_ref_agent_idx=0 \
    +output_json="$OUT_JSON"
