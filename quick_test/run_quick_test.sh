#!/bin/bash

# 多视图检索系统快速验证脚本
set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SMALL_DATASET_PATH="${BASE_DIR}/data/small_dataset"
LOG_DIR="${BASE_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG_FILE="${LOG_DIR}/quick_test_${TIMESTAMP}.log"

# 确保目录存在
mkdir -p "${LOG_DIR}"
mkdir -p "${SMALL_DATASET_PATH}"

# 日志函数
log() {
    local message="$1"
    local level="${2:-info}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [${level^^}] ${message}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [${level^^}] ${message}" >> "${MAIN_LOG_FILE}"
}

# 检查环境
check_environment() {
    log "检查Python环境..."
    if command -v python3 &> /dev/null; then
        log "Python版本: $(python3 --version 2>&1)"
        return 0
    else
        log "未找到Python3" "error"
        return 1
    fi
}

# 生成小数据集
generate_small_dataset() {
    log "生成小批量数据集..."
    if [ -f "${BASE_DIR}/small_dataset_generator.py" ]; then
        python3 "${BASE_DIR}/small_dataset_generator.py" \
            --source-dir "$(cat "${BASE_DIR}/config.json" | grep original_path | cut -d '"' -f 4)" \
            --target-dir "${SMALL_DATASET_PATH}" \
            --samples-per-class 10
        log "小批量数据集生成完成"
    else
        log "小批量数据集生成器不存在，跳过此步骤" "warn"
    fi
}

# 执行快速验证
run_quick_validation() {
    log "执行全流程快速验证..."
    if [ -f "${BASE_DIR}/quick_validation.py" ]; then
        python3 "${BASE_DIR}/quick_validation.py" \
            --dataset-path "${SMALL_DATASET_PATH}" \
            --output-dir "${BASE_DIR}"
        log "全流程验证完成"
        return 0
    else
        log "快速验证脚本不存在" "error"
        return 1
    fi
}

# 显示结果摘要
show_summary() {
    log "验证结果摘要:"
    LATEST_LOG=$(find "${LOG_DIR}" -name "quick_validation_*.log" -type f -mtime -1 | sort | tail -1)
    if [ -f "${LATEST_LOG}" ]; then
        log "详细日志: ${LATEST_LOG}"
        log "关键信息:"
        grep -E "Loss|召回率|特征统计|成功提取|错误计数" "${LATEST_LOG}" | tail -10
    else
        log "未找到验证日志文件" "warn"
    fi
    log "主日志文件: ${MAIN_LOG_FILE}"
}

# 主函数
main() {
    log "开始多视图检索系统快速验证流程"
    
    # 检查环境
    check_environment
    
    # 生成小数据集
    generate_small_dataset
    
    # 执行验证
    VALIDATION_SUCCESS=true
    if ! run_quick_validation; then
        VALIDATION_SUCCESS=false
    fi
    
    # 显示摘要
    show_summary
    
    # 输出结果
    echo "\n===================================="
    if $VALIDATION_SUCCESS; then
        echo "验证成功完成！"
    else
        echo "验证执行完成，但存在错误"
    fi
    echo "请查看日志文件了解详细结果"
    echo "===================================="
    
    return $([ "$VALIDATION_SUCCESS" = true ] && echo 0 || echo 1)
}

# 执行主函数
main