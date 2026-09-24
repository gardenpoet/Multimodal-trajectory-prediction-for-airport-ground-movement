#!/bin/bash
# KMSY H50, 1T checkpoint (single trajectory candidate per mode, sidestepping
# Contribution 2's K-selection question so this experiment isolates
# Contribution 3's MODE-level phenomenon). Mirrors run_find_cases_klax_50.sh.
#
# UNVERIFIED: unlike KLAX, no kmsy2_twophases_1_50.ckpt path has ever been
# referenced in any script in this repo (confirmed via a repo-wide grep) --
# this is inferred purely by analogy to KLAX's naming pattern. Run with
# LIMIT=5 first and check the .err log for a checkpoint-not-found error
# before trusting this at all.
#
# risk_assessment/ lives at the REPO ROOT (sibling to AmeliaTF_main,
# AmeliaTF_main_two_phases_4T, STGCNN_baseline, ...) -- submit from the repo
# root (e.g. /gpfs/scratch/exy064/ljx/Risk-Assessment/ on HPC):
#   sbatch --export=ALL,LIMIT=5 risk_assessment/run_find_cases_kmsy_50.sh

#SBATCH --job-name=amelia_risk_kmsy_50
#SBATCH --output=risk_kmsy_50_%j.out
#SBATCH --error=risk_kmsy_50_%j.err

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

OUT_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/out/risk_assessment/kmsy_50_cases.csv"
mkdir -p "$(dirname "$OUT_CSV")"

LIMIT_ARG=""
if [ -n "$LIMIT" ]; then
    LIMIT_ARG="+limit_batches=$LIMIT"
fi

python -m risk_assessment.find_cases_two_stage \
    ckpt=kmsy2 \
    data=kmsy.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +output_csv="$OUT_CSV" \
    $LIMIT_ARG
