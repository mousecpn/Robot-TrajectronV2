#!/usr/bin/env bash
# 多进程并行运行 BMI Navigation 数据采集脚本
# Usage: ./run_parallel_bmi.sh <num_processes> <total_data_per_proc> <data_dir>
# Example: ./run_parallel_bmi.sh 4 250000 ./data

# 参数解析
NUM_PROCS=${1:-4}
TOTAL_DATA=${2:-1000000}
DATA_DIR=${3:-"./data"}
SCRIPT="data_collection.py"  # 请根据实际脚本名称或路径修改

# 创建数据目录
mkdir -p "$DATA_DIR"

PER_PROC=$(( TOTAL_DATA / NUM_PROCS ))


# 启动子进程
for ((i=1; i<=NUM_PROCS; i++)); do
  DATA_PATH="$DATA_DIR/data.pkl"
  echo "[Process $i] Saving to $DATA_PATH, collecting $PER_PROC samples each (total $TOTAL_DATA)"
  python3 "$SCRIPT" \
    --total_data_num $PER_PROC \
    --data_path "$DATA_PATH" \
  &
done

# 等待所有子进程完成
wait

echo "All $NUM_PROCS processes finished."
