#!/bin/bash

# MVCNN预训练启动脚本

# 设置环境变量
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# 项目根目录
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")

# 创建必要的目录
mkdir -p "$PROJECT_ROOT/pre_checkpoints"
mkdir -p "$PROJECT_ROOT/logs/pre_train"

# 显示信息
echo "========================================"
echo "MVCNN预训练启动脚本"
echo "项目根目录: $PROJECT_ROOT"
echo "========================================"

# 检查Python环境
echo "检查Python环境..."
python --version

# 安装依赖（如果需要）
echo "安装必要的依赖..."
pip install -r "$PROJECT_ROOT/requirements.txt" 2>/dev/null || echo "requirements.txt不存在，跳过依赖安装"

# 运行预训练
echo "开始预训练..."
python "$PROJECT_ROOT/pre_training/pre_train.py" \
    --root_dir "/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0" \
    --batch_size 32 \
    --image_size 224 \
    --num_workers 4 \
    --feat_dim 512 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --checkpoint_dir "$PROJECT_ROOT/pre_checkpoints" \
    --log_dir "$PROJECT_ROOT/logs/pre_train" \
    --num_views 3 \
    --num_images_per_view 5

# 训练完成后的权重迁移（可选，取消注释以自动执行）
# echo "\n训练完成，开始权重迁移..."
# python "$PROJECT_ROOT/pre_training/transfer_weights.py" \
#     --pre_trained_path "$PROJECT_ROOT/pre_checkpoints/best_model.pth" \
#     --output_path "$PROJECT_ROOT/pre_checkpoints/transferred_encoder.pth" \
#     --create_full_model \
#     --full_model_output "$PROJECT_ROOT/pre_checkpoints/pretrained_main_model.pth"

echo "\n========================================"
echo "预训练脚本执行完成！"
echo "检查点保存位置: $PROJECT_ROOT/pre_checkpoints"
echo "日志保存位置: $PROJECT_ROOT/logs/pre_train"
echo "========================================"