#!/bin/bash

# View特征可视化脚本

echo "开始执行View级别特征可视化..." 

# 设置默认参数
FEATURE_DB_PATH="../features/feature_db.h5"
OUTPUT_DIR="../visualizations"
MAX_SAMPLES=1000
MAX_VIEWS_PER_OBJ=10

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --feature_db_path)
            FEATURE_DB_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --max_views_per_obj)
            MAX_VIEWS_PER_OBJ="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "使用以下参数:"
echo "  特征数据库路径: $FEATURE_DB_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  最大采样数量: $MAX_SAMPLES"
echo "  每个对象的最大view数量: $MAX_VIEWS_PER_OBJ"

# 运行View特征可视化脚本
python view_feature_visualization.py \
    --feature_db_path "$FEATURE_DB_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_samples $MAX_SAMPLES \
    --max_views_per_obj $MAX_VIEWS_PER_OBJ

echo "View级别特征可视化完成！"