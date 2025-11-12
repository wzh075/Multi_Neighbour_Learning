#!/bin/bash

# 多视图检索系统评估脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
DEFAULT_DATA_DIR="/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0"
DEFAULT_SPLIT="train"
DEFAULT_MODEL_PATH="../checkpoints/best_model.pth"
DEFAULT_MODEL_NAME="resnet50"
DEFAULT_FEATURE_DIM=1024
DEFAULT_BATCH_SIZE=32
DEFAULT_DEVICE="cuda"
DEFAULT_TOPK_LIST="1 5 10"
DEFAULT_OUTPUT_DIR="./evaluation_results"

# 帮助信息
show_help() {
    echo -e "${BLUE}用法: $0 [选项]${NC}"
    echo -e "\n选项:"
    echo -e "  ${YELLOW}--data-dir${NC}          数据集目录 (默认: ${DEFAULT_DATA_DIR})"
    echo -e "  ${YELLOW}--split${NC}             数据集划分 (默认: ${DEFAULT_SPLIT})"
    echo -e "  ${YELLOW}--model-path${NC}        预训练模型路径 (默认: ${DEFAULT_MODEL_PATH})"
    echo -e "  ${YELLOW}--model-name${NC}        模型名称 (默认: ${DEFAULT_MODEL_NAME})"
    echo -e "  ${YELLOW}--feature-dim${NC}       特征维度 (默认: ${DEFAULT_FEATURE_DIM})"
    echo -e "  ${YELLOW}--batch-size${NC}        批量大小 (默认: ${DEFAULT_BATCH_SIZE})"
    echo -e "  ${YELLOW}--device${NC}            运行设备 (默认: ${DEFAULT_DEVICE})"
    echo -e "  ${YELLOW}--topk-list${NC}         评估的top-k值列表 (默认: ${DEFAULT_TOPK_LIST})"
    echo -e "  ${YELLOW}--use-multiple-views${NC} 使用多视图检索 (默认: 禁用)"
    echo -e "  ${YELLOW}--output-dir${NC}        评估结果输出目录 (默认: ${DEFAULT_OUTPUT_DIR})"
    echo -e "  ${YELLOW}--debug${NC}             启用调试信息 (默认: 禁用)"
    echo -e "  ${YELLOW}-h, --help${NC}          显示帮助信息"
    echo -e "\n示例:"
    echo -e "  $0 --data-dir ../data --model-path ../models/best_model.pth --device cuda"
    echo -e "  $0 --data-dir ../data --split test --use-multiple-views --debug"
}

# 参数解析
parse_arguments() {
    DATA_DIR="${DEFAULT_DATA_DIR}"
    SPLIT="${DEFAULT_SPLIT}"
    MODEL_PATH="${DEFAULT_MODEL_PATH}"
    MODEL_NAME="${DEFAULT_MODEL_NAME}"
    FEATURE_DIM="${DEFAULT_FEATURE_DIM}"
    BATCH_SIZE="${DEFAULT_BATCH_SIZE}"
    DEVICE="${DEFAULT_DEVICE}"
    TOPK_LIST="${DEFAULT_TOPK_LIST}"
    USE_MULTIPLE_VIEWS=""
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
    DEBUG=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --split)
                SPLIT="$2"
                shift 2
                ;;
            --model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --model-name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --feature-dim)
                FEATURE_DIM="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --topk-list)
                TOPK_LIST="$2"
                shift 2
                ;;
            --use-multiple-views)
                USE_MULTIPLE_VIEWS="--use-multiple-views"
                shift 1
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --debug)
                DEBUG="--debug"
                shift 1
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
}

# 检查路径
check_paths() {
    # 检查数据集目录
    if [ ! -d "${DATA_DIR}" ]; then
        echo -e "${RED}错误: 数据集目录不存在: ${DATA_DIR}${NC}"
        exit 1
    fi
    
    # 检查模型文件
    if [ ! -f "${MODEL_PATH}" ]; then
        echo -e "${RED}错误: 模型文件不存在: ${MODEL_PATH}${NC}"
        exit 1
    fi
    
    # 创建输出目录
    mkdir -p "${OUTPUT_DIR}"
}

# 显示配置信息
show_config() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}           评估配置信息               ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${BLUE}数据集目录:${NC} ${DATA_DIR}"
    echo -e "${BLUE}数据集划分:${NC} ${SPLIT}"
    echo -e "${BLUE}模型路径:${NC} ${MODEL_PATH}"
    echo -e "${BLUE}模型名称:${NC} ${MODEL_NAME}"
    echo -e "${BLUE}特征维度:${NC} ${FEATURE_DIM}"
    echo -e "${BLUE}批量大小:${NC} ${BATCH_SIZE}"
    echo -e "${BLUE}运行设备:${NC} ${DEVICE}"
    echo -e "${BLUE}Top-K列表:${NC} ${TOPK_LIST}"
    echo -e "${BLUE}使用多视图检索:${NC} ${USE_MULTIPLE_VIEWS:+是}${USE_MULTIPLE_VIEWS:-否}"
    echo -e "${BLUE}输出目录:${NC} ${OUTPUT_DIR}"
    echo -e "${BLUE}日志文件:${NC} ${LOG_FILE}"
    echo -e "${BLUE}调试模式:${NC} ${DEBUG:+是}${DEBUG:-否}"
    echo -e "${GREEN}========================================${NC}"
}

# 主函数
main() {
    # 解析命令行参数
    parse_arguments "$@"
    
    # 生成日志文件路径
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${OUTPUT_DIR}/evaluation_${TIMESTAMP}.log"
    
    # 检查路径
    check_paths
    
    # 显示配置信息
    show_config
    
    # 构建评估命令
    EVALUATE_CMD="python evaluate.py \
        --data_dir "${DATA_DIR}" \
        --split "${SPLIT}" \
        --model_path "${MODEL_PATH}" \
        --model_name "${MODEL_NAME}" \
        --feature_dim "${FEATURE_DIM}" \
        --batch_size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --topk_list ${TOPK_LIST} \
        --output_dir "${OUTPUT_DIR}" \
        --log_file "${LOG_FILE}" \
        ${USE_MULTIPLE_VIEWS} \
        ${DEBUG}"
    
    echo -e "${BLUE}执行命令:${NC} ${EVALUATE_CMD}"
    echo -e "${YELLOW}评估开始时间: $(date)${NC}"
    echo -e "${YELLOW}日志将输出到: ${LOG_FILE}${NC}"
    
    # 执行评估
    eval "${EVALUATE_CMD}"
    
    # 检查执行状态
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}评估成功完成！${NC}"
        echo -e "${GREEN}评估结果和日志已保存到: ${OUTPUT_DIR}${NC}"
    else
        echo -e "${RED}评估失败，请查看日志文件了解详情。${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}评估结束时间: $(date)${NC}"
}

# 执行主函数
main "$@"