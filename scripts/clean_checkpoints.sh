#!/bin/bash

# 清理指定目录及其所有子目录下的 checkpoint-* 文件夹（递归查找）
# 用法: ./clean_checkpoints.sh <目录路径>

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供要清理的目录路径"
    echo "用法: $0 <目录路径>"
    echo "示例: $0 /path/to/your/model/directory"
    exit 1
fi

TARGET_DIR="$1"

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
    echo "错误: 目录 '$TARGET_DIR' 不存在"
    exit 1
fi

# 检查目录是否可写
if [ ! -w "$TARGET_DIR" ]; then
    echo "错误: 目录 '$TARGET_DIR' 不可写"
    exit 1
fi

echo "正在递归扫描目录: $TARGET_DIR"

# 查找所有 checkpoint-* 文件夹（递归查找所有子目录）
CHECKPOINT_DIRS=$(find "$TARGET_DIR" -type d -name "checkpoint-*" 2>/dev/null)

if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "未找到任何 checkpoint-* 文件夹"
    exit 0
fi

echo "找到以下 checkpoint 文件夹:"
echo "$CHECKPOINT_DIRS"
echo ""

# 计算总大小
TOTAL_SIZE=0
for dir in $CHECKPOINT_DIRS; do
    if [ -d "$dir" ]; then
        size=$(du -sb "$dir" 2>/dev/null | cut -f1)
        TOTAL_SIZE=$((TOTAL_SIZE + size))
    fi
done

# 显示总大小
if [ $TOTAL_SIZE -gt 0 ]; then
    if [ $TOTAL_SIZE -gt 1073741824 ]; then
        # 大于 1GB
        size_gb=$(echo "scale=2; $TOTAL_SIZE / 1073741824" | bc -l 2>/dev/null || echo "计算中...")
        echo "总大小: ${size_gb} GB"
    elif [ $TOTAL_SIZE -gt 1048576 ]; then
        # 大于 1MB
        size_mb=$(echo "scale=2; $TOTAL_SIZE / 1048576" | bc -l 2>/dev/null || echo "计算中...")
        echo "总大小: ${size_mb} MB"
    else
        echo "总大小: $TOTAL_SIZE 字节"
    fi
fi

echo ""

# 确认删除
read -p "确定要删除这些文件夹吗? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

# 执行删除
echo "开始删除 checkpoint 文件夹..."
deleted_count=0
for dir in $CHECKPOINT_DIRS; do
    if [ -d "$dir" ]; then
        echo "删除: $dir"
        if rm -rf "$dir" 2>/dev/null; then
            deleted_count=$((deleted_count + 1))
            echo "  ✓ 删除成功"
        else
            echo "  ✗ 删除失败"
        fi
    fi
done

echo ""
echo "清理完成! 共删除了 $deleted_count 个 checkpoint 文件夹"
