#!/bin/bash
# Ambiguity-gated illustrative case for KBOS (batch=300, sample=35,
# ego_id/ref agent auto-detected). Tried after b359/s79, b379/s98,
# b187/s114, and b373/s25 (all genuinely ambiguous mode-error cases with
# a good risk story, but each ended up visually reducing to just two
# clearly-visible candidate lines once per-mode opacity correctly
# scales with mode_prob -- the third feasible mode's probability was
# always negligible). This one comes from kbos_50_4T_cases_ranked.csv's
# own ambiguous=True column, filtered specifically for all FOUR modes
# being topologically feasible (feasible_modes has 4 entries, not the
# usual 2-3), on the reasoning that a topology unconstrained by the
# turn-feasibility mask is more likely to spread real probability mass
# across more than two modes: argmax_mode=TurnRight vs gt_mode=Straight
# (a genuine mode_error), realized outcome safe (true_min_sep_gt~132.8m)
# but under the old 50m mean-only metric risk_naive=0 (>=50m separation)
# while risk_gated==risk_worst_case~48.7/1000 (~1.3m, near-collision):
# gating fully recovers a severe danger that naive completely misses. 4T
# checkpoint -- same 1T/2T/4T treatment as the other cases, for
# consistency.
#
# Run from the repo root:
#   sbatch risk_assessment/run_case_risk_dynamics_kbos_groupb_b300s35_4T.sh

#SBATCH --job-name=case_risk_dynamics_kbos_groupb_b300s35_4T
#SBATCH --output=risk_assessment/case_risk_dynamics_kbos_groupb_b300s35_4T_%j.out
#SBATCH --error=risk_assessment/case_risk_dynamics_kbos_groupb_b300s35_4T_%j.err

#SBATCH --partition=compute

#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00

cd $SLURM_SUBMIT_DIR

module load miniforge/25.3.0
conda activate amelia_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HYDRA_FULL_ERROR=1

OUT_JSON="/gpfs/scratch/exy064/ljx/Risk-Assessment/risk_assessment/out/kbos_case_300_35_groupb_dynamics_4T.json"
mkdir -p "$(dirname "$OUT_JSON")"

python -m risk_assessment.case_risk_dynamics \
    ckpt=kbos2 \
    +data.dataset.config.random_ego=false \
    data=kbos.yaml \
    mode_ckpt_path='${ckpt_dir}/${type}/${ckpt}/mode_model/${ckpt}_twophases_50.ckpt' \
    traj_ckpt_path='${ckpt_dir}/${type}/${ckpt}/traj_model/${ckpt}_twophases_4_50.ckpt' \
    model.traj_net.config.num_hypotheses=4 \
    +model.traj_net.config.decoder.enable_score_head=true \
    +model.traj_net.config.decoder.score_mode=5 \
    +model.traj_net.config.decoder.score_head_type=attention \
    +scorer.score_head_load='/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main_two_phases_4T/out/${ckpt}/per_mode_scorer_hard.pt' \
    +case_batch_idx=300 +case_sample_idx=35 \
    +output_json="$OUT_JSON"
