#!/bin/bash
# -*- coding: utf-8 -*-

# 多视图检索系统 - 验证脚本
# Bash Shell 版本

# 设置默认参数
ROOT_DIR="/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0"
SPLIT="train"
NUM_VIEWS=3
NUM_IMAGES_PER_VIEW=5
IMAGE_SIZE=224
MODEL_PATH="../checkpoints/best_model.pth"
FEAT_DIM=512
NUM_CLASSES=40
FEATURE_DB_PATH="../features/feature_db.h5"
OUTPUT_DIR="./results"

# 打印脚本信息
echo -e "\033[36m"
echo "========================================================="
echo "                  多视图检索系统 - 验证脚本                "
echo "========================================================="
echo -e "\033[0m"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --root_dir)
            ROOT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num_views)
            NUM_VIEWS="$2"
            shift 2
            ;;
        --num_images_per_view)
            NUM_IMAGES_PER_VIEW="$2"
            shift 2
            ;;
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --feat_dim)
            FEAT_DIM="$2"
            shift 2
            ;;
        --num_classes)
            NUM_CLASSES="$2"
            shift 2
            ;;
        --feature_db_path)
            FEATURE_DB_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --root_dir <路径>            数据集根目录 (默认: $ROOT_DIR)"
            echo "  --split <train/val/test>     数据集分割 (默认: $SPLIT)"
            echo "  --num_views <数量>           视点组数量 (默认: $NUM_VIEWS)"
            echo "  --num_images_per_view <数量> 每个视点组图像数 (默认: $NUM_IMAGES_PER_VIEW)"
            echo "  --image_size <尺寸>          图像尺寸 (默认: $IMAGE_SIZE)"
            echo "  --model_path <路径>          模型文件路径 (必填)"
            echo "  --feat_dim <维度>            特征维度 (默认: $FEAT_DIM)"
            echo "  --num_classes <数量>         类别数量 (默认: $NUM_CLASSES)"
            echo "  --feature_db_path <路径>     特征数据库路径 (必填)"
            echo "  --output_dir <路径>          结果输出目录 (默认: $OUTPUT_DIR)"
            echo "  -h, --help                   显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [[ -z "$MODEL_PATH" ]]; then
    echo -e "\033[31m错误: 必须指定模型文件路径 --model_path\033[0m"
    exit 1
fi

if [[ -z "$FEATURE_DB_PATH" ]]; then
    echo -e "\033[31m错误: 必须指定特征数据库路径 --feature_db_path\033[0m"
    exit 1
fi

# 检查文件是否存在
if [[ ! -f "$MODEL_PATH" ]]; then
    echo -e "\033[31m错误: 模型文件不存在: $MODEL_PATH\033[0m"
    exit 1
fi

if [[ ! -f "$FEATURE_DB_PATH" ]]; then
    echo -e "\033[31m错误: 特征数据库文件不存在: $FEATURE_DB_PATH\033[0m"
    exit 1
fi

# 打印配置信息
echo -e "\033[33m[配置信息]\033[0m"
echo -e "\033[33m数据集路径: $ROOT_DIR\033[0m"
echo -e "\033[33m数据集分割: $SPLIT\033[0m"
echo -e "\033[33m视点组数量: $NUM_VIEWS\033[0m"
echo -e "\033[33m每个视点组图像数: $NUM_IMAGES_PER_VIEW\033[0m"
echo -e "\033[33m图像尺寸: $IMAGE_SIZE\033[0m"
echo -e "\033[33m模型路径: $MODEL_PATH\033[0m"
echo -e "\033[33m特征维度: $FEAT_DIM\033[0m"
echo -e "\033[33m类别数量: $NUM_CLASSES\033[0m"
echo -e "\033[33m特征数据库: $FEATURE_DB_PATH\033[0m"
echo -e "\033[33m输出目录: $OUTPUT_DIR\033[0m"
echo

# 创建输出目录
if [[ ! -d "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    echo -e "\033[32m创建输出目录: $OUTPUT_DIR\033[0m"
fi

# 构建命令行参数
CMD_ARGS="--root_dir \"$ROOT_DIR\" \
           --split $SPLIT \
           --num_views $NUM_VIEWS \
           --num_images_per_view $NUM_IMAGES_PER_VIEW \
           --image_size $IMAGE_SIZE \
           --model_path \"$MODEL_PATH\" \
           --feat_dim $FEAT_DIM \
           --num_classes $NUM_CLASSES \
           --feature_db_path \"$FEATURE_DB_PATH\" \
           --output_dir \"$OUTPUT_DIR\""

# 调用验证脚本
echo -e "\033[32m开始验证...\033[0m"
echo

# 使用 Python 执行验证代码
PYTHON_CMD="python evaluate.py $CMD_ARGS"
echo -e "\033[90m执行命令: $PYTHON_CMD\033[0m"
echo

# 执行命令
python evaluate.py \
    --root_dir "$ROOT_DIR" \
    --split "$SPLIT" \
    --num_views "$NUM_VIEWS" \
    --num_images_per_view "$NUM_IMAGES_PER_VIEW" \
    --image_size "$IMAGE_SIZE" \
    --model_path "$MODEL_PATH" \
    --feat_dim "$FEAT_DIM" \
    --num_classes "$NUM_CLASSES" \
    --feature_db_path "$FEATURE_DB_PATH" \
    --output_dir "$OUTPUT_DIR"

# 检查执行结果
if [[ $? -eq 0 ]]; then
    echo
    echo -e "\033[32m=========================================================\033[0m"
    echo -e "\033[32m                  验证完成，结果已保存！                  \033[0m"
    echo -e "\033[32m=========================================================\033[0m"
    echo 
    echo -e "\033[32m结果文件路径: $OUTPUT_DIR/evaluation_results.json\033[0m"
else
    echo
    echo -e "\033[31m=========================================================\033[0m"
    echo -e "\033[31m                   验证失败！                           \033[0m"
    echo -e "\033[31m=========================================================\033[0m"
    echo 
    echo -e "\033[31m错误代码: $?\033[0m"
    exit 1
fi
