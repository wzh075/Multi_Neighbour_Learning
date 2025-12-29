# View特征可视化脚本 (PowerShell)

Write-Host "开始执行View级别特征可视化..." -ForegroundColor Green

# 设置默认参数
$feature_db_path = "../features/feature_db.h5"
$output_dir = "../visualizations"
$max_samples = 1000
$max_views_per_obj = 10

# 检查命令行参数
for ($i = 0; $i -lt $args.Count; $i += 2) {
    if ($i + 1 -lt $args.Count) {
        switch ($args[$i]) {
            "--feature_db_path" { $feature_db_path = $args[$i + 1]; break }
            "--output_dir" { $output_dir = $args[$i + 1]; break }
            "--max_samples" { $max_samples = [int]$args[$i + 1]; break }
            "--max_views_per_obj" { $max_views_per_obj = [int]$args[$i + 1]; break }
        }
    }
}

Write-Host "使用以下参数:" -ForegroundColor Yellow
Write-Host "  特征数据库路径: $feature_db_path"
Write-Host "  输出目录: $output_dir"
Write-Host "  最大采样数量: $max_samples"
Write-Host "  每个对象的最大view数量: $max_views_per_obj"

# 运行View特征可视化脚本
Write-Host "执行命令: python view_feature_visualization.py --feature_db_path '$feature_db_path' --output_dir '$output_dir' --max_samples $max_samples --max_views_per_obj $max_views_per_obj" -ForegroundColor Cyan

python view_feature_visualization.py --feature_db_path "$feature_db_path" --output_dir "$output_dir" --max_samples $max_samples --max_views_per_obj $max_views_per_obj

Write-Host "View级别特征可视化完成！" -ForegroundColor Green