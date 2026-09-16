#!/bin/bash
#
# reset_mtime.sh
# 递归地将指定文件夹下所有文件和子目录的修改时间重置为当前时间
#
# 用法:
#   ./reset_mtime.sh                # 使用脚本内默认的4个目录
#   ./reset_mtime.sh dir1 dir2 ...  # 指定要处理的目录
#
# 说明:
#   - 使用 touch 命令将文件的 mtime(和 atime)更新为当前时间
#   - 递归处理所有子目录、子文件
#   - 若目录不存在会给出提示并跳过

set -euo pipefail

# 默认目录列表(根据截图中的4个文件夹,按需修改路径)
DEFAULT_DIRS=(
    "kbos"
    "blacklist"
    "kmsy"
    "klax"
)

# 如果传入了参数,则使用参数作为目录列表;否则使用默认列表
if [ "$#" -gt 0 ]; then
    DIRS=("$@")
else
    DIRS=("${DEFAULT_DIRS[@]}")
fi

total_count=0

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "警告: 目录不存在,跳过 -> $dir"
        continue
    fi

    echo "正在处理目录: $dir"

    # 统计文件数量(不含目录本身)
    count=$(find "$dir" -type f | wc -l)

    # 递归 touch 所有文件和子目录
    find "$dir" -exec touch {} +

    echo "  已更新 $count 个文件的修改时间"
    total_count=$((total_count + count))
done

echo "完成。共更新 $total_count 个文件。"
