#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=201, sample=103,
# ego_id/ref agent auto-detected). Tried after four earlier picks that
# each visually reduced to only two clearly-visible candidate lines;
# see run_case_risk_dynamics_kbos_groupb_b201s103_4T.sh for the full
# rationale (filtered for all four modes topologically feasible, on
# the reasoning that leaves more room for real probability mass to
# spread beyond the top two). argmax_mode=TurnRight vs gt_mode=Straight
# (a genuine mode_error), with gated fully recovering worst-case's much
# higher risk that naive alone misses, under the old 50m mean-only
# metric. 1T checkpoint --
# same 1T/2T/4T treatment as the other cases, for consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b201s103_1T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b201s103_1T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b201s103_1T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b201s103_1T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_201_103_groupb_dynamics_1T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_1_50.ckpt' \
    model.traj_net.config.num_hypotheses=1 \
    +case_batch_idx=201 +case_sample_idx=103 \
    +output_json="$OUT_JSON"
