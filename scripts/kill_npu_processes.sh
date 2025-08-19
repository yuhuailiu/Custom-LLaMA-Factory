#!/bin/bash

# 脚本名称: kill_npu_processes.sh
# 功能: 一键kill掉npu-smi info中占用NPU的所有进程
# 作者: Auto-generated
# 使用方法: ./kill_npu_processes.sh [选项]
#   -f, --force    强制kill所有NPU进程（使用kill -9）
#   -h, --help     显示帮助信息
#   -y, --yes      自动确认，不询问用户

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
FORCE_KILL=false
AUTO_CONFIRM=false

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -f, --force    强制kill所有NPU进程（使用kill -9）"
    echo "  -y, --yes      自动确认，不询问用户"
    echo "  -h, --help     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0              # 正常模式，会询问确认"
    echo "  $0 -y           # 自动确认模式"
    echo "  $0 -f -y        # 强制kill模式，自动确认"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE_KILL=true
            shift
            ;;
        -y|--yes)
            AUTO_CONFIRM=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 '$1'${NC}"
            show_help
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}     NPU 进程清理脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查npu-smi命令是否存在
if ! command -v npu-smi &> /dev/null; then
    echo -e "${RED}错误: npu-smi 命令未找到！${NC}"
    echo "请确保华为昇腾NPU驱动已正确安装。"
    exit 1
fi

echo -e "${YELLOW}正在扫描NPU使用情况...${NC}"

# 获取npu-smi info输出
NPU_INFO=$(npu-smi info 2>/dev/null)

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 无法执行 npu-smi info 命令${NC}"
    exit 1
fi

# 提取进程ID（从Process id列中提取，排除表头）
PROCESS_IDS=$(echo "$NPU_INFO" | grep -E "^\|.*\|.*\|.*\|.*\|$" | grep -v "Process id" | grep -v "==="| awk -F'|' '{print $3}' | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' | tr -d ' ' | sort -u)

if [ -z "$PROCESS_IDS" ]; then
    echo -e "${GREEN}✓ 没有发现占用NPU的进程${NC}"
    exit 0
fi

# 显示找到的进程
echo -e "${YELLOW}发现以下进程正在占用NPU:${NC}"
echo

# 创建一个临时文件来存储进程信息
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

for PID in $PROCESS_IDS; do
    # 检查进程是否仍然存在
    if kill -0 $PID 2>/dev/null; then
        # 获取进程详细信息
        PROCESS_INFO=$(ps -p $PID -o pid,ppid,user,cmd --no-headers 2>/dev/null || echo "$PID ? ? [进程已退出]")
        echo -e "${RED}PID: $PID${NC} - $PROCESS_INFO"
        echo $PID >> $TEMP_FILE
    else
        echo -e "${YELLOW}PID: $PID - [进程已退出]${NC}"
    fi
done

# 读取仍然存在的进程ID
ACTIVE_PIDS=$(cat $TEMP_FILE 2>/dev/null)

if [ -z "$ACTIVE_PIDS" ]; then
    echo -e "${GREEN}✓ 所有NPU进程都已退出${NC}"
    exit 0
fi

echo
ACTIVE_COUNT=$(echo "$ACTIVE_PIDS" | wc -l)
echo -e "${YELLOW}共找到 $ACTIVE_COUNT 个活动的NPU进程${NC}"

# 用户确认
if [ "$AUTO_CONFIRM" = false ]; then
    echo
    if [ "$FORCE_KILL" = true ]; then
        echo -e "${RED}警告: 将使用 kill -9 强制终止这些进程！${NC}"
    else
        echo -e "${YELLOW}将使用 kill -15 优雅地终止这些进程${NC}"
    fi
    
    read -p "确定要继续吗? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}操作已取消${NC}"
        exit 0
    fi
fi

# Kill进程
echo
echo -e "${YELLOW}正在终止NPU进程...${NC}"

SUCCESS_COUNT=0
FAILED_COUNT=0

for PID in $ACTIVE_PIDS; do
    if [ "$FORCE_KILL" = true ]; then
        KILL_CMD="kill -9 $PID"
        SIGNAL="SIGKILL"
    else
        KILL_CMD="kill -15 $PID"  
        SIGNAL="SIGTERM"
    fi
    
    if $KILL_CMD 2>/dev/null; then
        echo -e "${GREEN}✓ 已发送 $SIGNAL 信号给进程 $PID${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo -e "${RED}✗ 无法终止进程 $PID (可能权限不足或进程已退出)${NC}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# 等待进程终止 (仅在非强制模式下)
if [ "$FORCE_KILL" = false ] && [ $SUCCESS_COUNT -gt 0 ]; then
    echo -e "${YELLOW}等待进程优雅退出...${NC}"
    sleep 3
    
    # 检查是否还有未退出的进程
    REMAINING_PIDS=""
    for PID in $ACTIVE_PIDS; do
        if kill -0 $PID 2>/dev/null; then
            REMAINING_PIDS="$REMAINING_PIDS $PID"
        fi
    done
    
    if [ -n "$REMAINING_PIDS" ]; then
        echo -e "${YELLOW}以下进程仍在运行，将强制终止:${NC}"
        for PID in $REMAINING_PIDS; do
            echo -e "${YELLOW}强制终止进程 $PID${NC}"
            kill -9 $PID 2>/dev/null || echo -e "${RED}无法强制终止进程 $PID${NC}"
        done
    fi
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ 进程清理完成${NC}"
echo -e "${GREEN}  成功: $SUCCESS_COUNT${NC}"
if [ $FAILED_COUNT -gt 0 ]; then
    echo -e "${RED}  失败: $FAILED_COUNT${NC}"
fi
echo -e "${BLUE}========================================${NC}"

# 最终检查
echo -e "${YELLOW}正在验证NPU状态...${NC}"
sleep 1
FINAL_CHECK=$(npu-smi info 2>/dev/null | grep -E "^\|.*\|.*\|.*\|.*\|$" | grep -v "Process id" | grep -v "===" | awk -F'|' '{print $3}' | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' | tr -d ' ' | sort -u)

if [ -z "$FINAL_CHECK" ]; then
    echo -e "${GREEN}✓ NPU已清空，所有进程已成功终止${NC}"
else
    echo -e "${YELLOW}⚠ 仍有进程在使用NPU:${NC}"
    for PID in $FINAL_CHECK; do
        PROCESS_INFO=$(ps -p $PID -o pid,user,cmd --no-headers 2>/dev/null || echo "$PID [未知进程]")
        echo -e "${YELLOW}  $PROCESS_INFO${NC}"
    done
fi
