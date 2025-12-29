# 特征可视化脚本 (PowerShell)

# 设置参数
$FEATURE_DB_PATH = "..\features\feature_db.h5"  # 特征数据库路径
$FEAT_TYPE = "obj_feat"                        # 特征类型: obj_feat, global_features, view_features
$OUTPUT_DIR = "..\visualizations"              # 输出目录
$MAX_SAMPLES = 1000                            # 最大采样数量（如果特征太多，可减少计算时间）

# 创建输出目录
if (!(Test-Path $OUTPUT_DIR)) {
    New-Item -ItemType Directory -Path $OUTPUT_DIR -Force
}

Write-Host "开始可视化Object级别特征分布..."
Write-Host "特征数据库: $FEATURE_DB_PATH"
Write-Host "特征类型: $FEAT_TYPE"
Write-Host "输出目录: $OUTPUT_DIR"
Write-Host "最大采样数量: $MAX_SAMPLES"

# 运行可视化脚本
python object_feature_visualization.py --feature_db_path $FEATURE_DB_PATH --feat_type $FEAT_TYPE --output_dir $OUTPUT_DIR --max_samples $MAX_SAMPLES

Write-Host "特征可视化完成！"
Write-Host "结果已保存至: $OUTPUT_DIR"