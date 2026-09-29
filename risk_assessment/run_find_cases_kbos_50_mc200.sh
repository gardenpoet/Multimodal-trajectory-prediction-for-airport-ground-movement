#!/bin/bash
# KBOS H50, 1T checkpoint (single trajectory candidate per mode, sidestepping
# Contribution 2's K-selection question so this experiment isolates
# Contribution 3's MODE-level phenomenon). Mirrors run_find_cases_klax_50.sh.
#
# UNVERIFIED: unlike KLAX, no kbos2_twophases_1_50.ckpt path has ever been
# referenced in any script in this repo (confirmed via a repo-wide grep) --
# this is inferred purely by analogy to KLAX's naming pattern. Run with
# LIMIT=5 first and check the .err log for a checkpoint-not-found error
# before trusting this at all.
#
# risk_assessment/ lives at the REPO ROOT (sibling to AmeliaTF_main,
# AmeliaTF_main_two_phases_4T, STGCNN_baseline, ...) -- submit from the repo
# root (e.g. /gpfs/scratch/exy064/ljx/Risk-Assessment/ on HPC):
#   sbatch --export=ALL,LIMIT=5 risk_assessment/run_find_cases_kbos_50_mc200.sh

#SBATCH --job-name=amelia_risk_kbos_50_mc200
#SBATCH --output=risk_assessment/risk_kbos_50_mc200_%j.out
#SBATCH --error=risk_assessment/risk_kbos_50_mc200_%j.err

#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=48:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_CSV="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_50_mc200_cases.csv"
mkdir -p "$(dirname "$OUT_CSV")"

# Defaults to a ~1/10 random subsample: prepare_data() shuffles the scene
# file list with a fixed seed BEFORE assigning batch indices, so a
# batch-index prefix is a genuine random subsample, not a biased "first N
# files" slice. Override with --export=ALL,LIMIT=<batches> for a full run
# (KBOS test set: ~499249 scenes / 128 batch_size =~ 3901 batches) or a
# quick sanity check (e.g. LIMIT=5).
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="+limit_batches=${LIMIT}"

python -m risk_assessment.find_cases_two_stage \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +risk_method=mc \
    +mc_threshold_ft=200 \
    +mc_samples=100 \
    +output_csv="$OUT_CSV" \
    $LIMIT_ARG
