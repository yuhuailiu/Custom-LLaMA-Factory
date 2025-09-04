#!/bin/bash

# 数据同步脚本
# 用于从远程服务器同步data文件夹到本地

# 配置参数
REMOTE_HOST="root@115.120.29.0"
REMOTE_PATH="/data2/lyh/Custom-LLaMA-Factory/data"
LOCAL_PATH="/data/k8s/lyh/Custom-LLaMA-Factory/data"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -d, --dry-run  仅显示将要执行的操作，不实际同步"
    echo "  -v, --verbose  显示详细输出"
    echo "  --delete       删除目标中源没有的文件（危险操作）"
    echo ""
    echo "示例:"
    echo "  $0                    # 执行同步"
    echo "  $0 -d                 # 预览同步操作"
    echo "  $0 -v                 # 详细输出同步"
    echo "  $0 --delete           # 同步并删除多余文件"
}

# 检查rsync是否安装
check_rsync() {
    if ! command -v rsync &> /dev/null; then
        log_error "rsync 未安装，请先安装 rsync"
        echo "Ubuntu/Debian: sudo apt-get install rsync"
        echo "CentOS/RHEL: sudo yum install rsync"
        exit 1
    fi
}

# 检查SSH连接
check_ssh_connection() {
    log_info "检查SSH连接..."
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes $REMOTE_HOST "echo 'SSH连接成功'" &> /dev/null; then
        log_error "无法连接到远程主机 $REMOTE_HOST"
        log_error "请确保："
        log_error "1. SSH密钥已正确配置"
        log_error "2. 远程主机可访问"
        log_error "3. 用户有访问权限"
        exit 1
    fi
    log_info "SSH连接正常"
}

# 检查远程路径是否存在
check_remote_path() {
    log_info "检查远程路径是否存在..."
    if ! ssh $REMOTE_HOST "test -d $REMOTE_PATH"; then
        log_error "远程路径 $REMOTE_PATH 不存在"
        exit 1
    fi
    log_info "远程路径存在"
}

# 创建本地目录
create_local_dir() {
    if [ ! -d "$LOCAL_PATH" ]; then
        log_info "创建本地目录: $LOCAL_PATH"
        mkdir -p "$LOCAL_PATH"
    fi
}

# 执行同步
sync_data() {
    local dry_run=$1
    local verbose=$2
    local delete=$3
    
    # 构建rsync命令
    local rsync_cmd="rsync -avz"
    
    if [ "$dry_run" = true ]; then
        rsync_cmd="$rsync_cmd --dry-run"
        log_info "预览模式：将显示将要执行的操作"
    fi
    
    if [ "$verbose" = true ]; then
        rsync_cmd="$rsync_cmd --progress"
    fi
    
    if [ "$delete" = true ]; then
        rsync_cmd="$rsync_cmd --delete"
        log_warn "启用删除模式：将删除目标中源没有的文件"
    fi
    
    # 添加排除规则（可选）
    rsync_cmd="$rsync_cmd --exclude='*.tmp' --exclude='*.log' --exclude='.DS_Store'"
    
    # 执行同步
    log_info "开始同步数据..."
    log_info "源: $REMOTE_HOST:$REMOTE_PATH"
    log_info "目标: $LOCAL_PATH"
    
    if [ "$dry_run" = true ]; then
        echo ""
        echo "将要执行的命令:"
        echo "$rsync_cmd $REMOTE_HOST:$REMOTE_PATH/ $LOCAL_PATH/"
        echo ""
    fi
    
    if $rsync_cmd "$REMOTE_HOST:$REMOTE_PATH/" "$LOCAL_PATH/"; then
        log_info "数据同步完成！"
    else
        log_error "数据同步失败！"
        exit 1
    fi
}

# 显示同步统计信息
show_stats() {
    if [ -d "$LOCAL_PATH" ]; then
        local size=$(du -sh "$LOCAL_PATH" 2>/dev/null | cut -f1)
        local files=$(find "$LOCAL_PATH" -type f | wc -l)
        log_info "本地数据统计:"
        log_info "  路径: $LOCAL_PATH"
        log_info "  大小: $size"
        log_info "  文件数: $files"
    fi
}

# 主函数
main() {
    local dry_run=false
    local verbose=false
    local delete=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dry-run)
                dry_run=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            --delete)
                delete=true
                shift
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 显示脚本信息
    echo "=========================================="
    echo "    数据同步脚本"
    echo "=========================================="
    echo "远程主机: $REMOTE_HOST"
    echo "远程路径: $REMOTE_PATH"
    echo "本地路径: $LOCAL_PATH"
    echo "=========================================="
    echo ""
    
    # 执行检查
    check_rsync
    check_ssh_connection
    check_remote_path
    create_local_dir
    
    # 执行同步
    sync_data $dry_run $verbose $delete
    
    # 显示统计信息
    if [ "$dry_run" = false ]; then
        show_stats
    fi
    
    log_info "脚本执行完成！"
}

# 捕获中断信号
trap 'log_error "脚本被中断"; exit 1' INT TERM

# 执行主函数
main "$@"
