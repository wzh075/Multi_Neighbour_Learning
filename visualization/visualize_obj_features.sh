#!/bin/bash
# 特征可视化脚本

# 设置参数
FEATURE_DB_PATH="../features/feature_db.h5"  # 特征数据库路径
FEAT_TYPE="obj_feat"                        # 特征类型: obj_feat, global_features, view_features
OUTPUT_DIR="../visualizations"              # 输出目录
MAX_SAMPLES=1000                            # 最大采样数量（如果特征太多，可减少计算时间）

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "开始可视化Object级别特征分布..."
echo "特征数据库: $FEATURE_DB_PATH"
echo "特征类型: $FEAT_TYPE"
echo "输出目录: $OUTPUT_DIR"
echo "最大采样数量: $MAX_SAMPLES"

# 运行可视化脚本
python object_feature_visualization.py \
    --feature_db_path $FEATURE_DB_PATH \
    --feat_type $FEAT_TYPE \
    --output_dir $OUTPUT_DIR \
    --max_samples $MAX_SAMPLES

echo "特征可视化完成！"
echo "结果已保存至: $OUTPUT_DIR"