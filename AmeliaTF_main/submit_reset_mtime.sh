#!/bin/bash
#SBATCH --job-name=reset_mtime          # 作业名称
#SBATCH --output=reset_mtime_%j.log     # 标准输出日志(%j 会替换成作业ID)
#SBATCH --error=reset_mtime_%j.err      # 错误日志
#SBATCH --time=12:00:00                 # 最长运行时间 (时:分:秒),按需调整
#SBATCH --ntasks=1                      # 单任务
#SBATCH --cpus-per-task=4               # 单核即可,touch 是 IO 密集型操作
#SBATCH --mem=20G                        # 内存需求,足够小任务用

# ------------------------------------------------------------------
# submit_reset_mtime.sh
# 提交到 Slurm 调度系统,在计算节点上运行 reset_mtime.sh
#
# 用法:
#   sbatch submit_reset_mtime.sh
#   (确保 reset_mtime.sh 与本脚本在同一目录下,或修改下面的路径)
#
# 提交后可用以下命令查看状态:
#   squeue -u $USER
# 查看日志:
#   tail -f reset_mtime_<job_id>.log
# ------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# 要处理的目标目录(改成你实际的绝对路径)
BASE_DIR="/gpfs/scratch/exy064/ljx/Risk-Assessment/AmeliaTF_main/datasets/amelia/traj_data_a10v08/proc_full_scenes3"

echo "作业开始时间: $(date)"
echo "运行节点: $(hostname)"
echo "处理目录: $BASE_DIR"

cd "$BASE_DIR"

bash "$SCRIPT_DIR/reset_mtime.sh" kbos blacklist kmsy klax

echo "作业结束时间: $(date)"
