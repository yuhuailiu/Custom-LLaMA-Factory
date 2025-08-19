#!/bin/bash

# 简化版NPU进程清理脚本
# 使用方法: ./kill_npu_simple.sh

echo "正在清理NPU进程..."

# 获取所有占用NPU的进程ID
PIDS=$(npu-smi info 2>/dev/null | grep -E "^\|.*\|.*\|.*\|.*\|$" | grep -v "Process id" | grep -v "===" | awk -F'|' '{print $3}' | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' | tr -d ' ' | sort -u)

if [ -z "$PIDS" ]; then
    echo "✓ 没有发现占用NPU的进程"
    exit 0
fi

echo "发现NPU进程: $PIDS"

# Kill所有进程
for PID in $PIDS; do
    if kill -0 $PID 2>/dev/null; then
        echo "正在终止进程 $PID..."
        kill -15 $PID 2>/dev/null || kill -9 $PID 2>/dev/null
    fi
done

# 等待并检查
sleep 2
echo "✓ NPU进程清理完成"
