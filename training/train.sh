#!/bin/bash

# 多视图检索系统训练脚本

echo "===== 多视图检索系统训练启动 ====="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1  # 使用指定的GPU设备

# 进入脚本所在目录
cd "$(dirname "$0")"

# 激活虚拟环境（如果需要）
# source /path/to/your/venv/bin/activate

# 设置训练参数
ROOT_DIR="/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0"  # 数据集路径
BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=0.0001
SAVE_DIR="../checkpoints"
FEAT_DIM=512
IMAGE_SIZE=224
NUM_VIEWS=3
NUM_IMAGES_PER_VIEW=5
PRETRAINED_WEIGHTS="/data1/Wuzhihe/Multi_Neighbour_Learning_v1.0/pre_checkpoints/best_feature_extractor.pth"  # 预训练特征提取器权重路径
FREEZE_ENCODER=false  # 是否冻结编码器

# 创建保存目录
mkdir -p "$SAVE_DIR"

echo "开始训练模型..."
echo "数据集路径: $ROOT_DIR"
echo "批大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LEARNING_RATE"
echo "保存目录: $SAVE_DIR"
echo "预训练权重: $PRETRAINED_WEIGHTS"
echo "冻结编码器: $FREEZE_ENCODER"
echo ""

# 运行训练脚本
python main.py \
    --root_dir "$ROOT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --save_dir "$SAVE_DIR" \
    --feat_dim "$FEAT_DIM" \
    --image_size "$IMAGE_SIZE" \
    --num_views "$NUM_VIEWS" \
    --num_images_per_view "$NUM_IMAGES_PER_VIEW" \
    --tau 0.07 \
    --lambda_view_sim 0.1 \
    --pretrained_weights "$PRETRAINED_WEIGHTS" \
    $([ "$FREEZE_ENCODER" = true ] && echo "--freeze_encoder" || echo "") \
    --lambda_global_consistency 0.1 \
    --feat_reg_weight 0.5 \
    --feat_activation_scaling 2.0

if [ $? -eq 0 ]; then
    echo ""
    echo "===== 训练完成！模型已保存至 $SAVE_DIR ====="
else
    echo ""
    echo "===== 训练失败！请检查错误信息 ====="
    exit 1
fi