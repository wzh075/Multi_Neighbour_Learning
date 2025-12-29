  #!/bin/bash

# 多视图检索系统 - 特征提取脚本

echo "多视图检索系统 - 特征提取脚本"
echo "================================"

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $(readlink -f $0)))

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# 默认参数
ROOT_DIR="/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0"
SPLIT="test"
FEATURE_DB="../features/feature_db.h5"
MODEL_PATH="../checkpoints/best_model.pth"
BATCH_SIZE=16
NUM_WORKERS=4
VERBOSE=false
IMAGE_SIZE=224
FEAT_DIM=512
NUM_VIEWS=3
NUM_IMAGES_PER_VIEW=5

# 显示帮助信息
show_help() {
    echo -e "\n使用方法: $0 [选项]"
    echo -e "\n选项:"
    echo -e "  -h, --help                显示此帮助信息"
    echo -e "  --root_dir <目录>         数据集根目录 (默认: $ROOT_DIR)"
    echo -e "  --split <train/val/test>  数据集分割 (默认: $SPLIT)"
    echo -e "  --feature_db <路径>       特征数据库保存路径 (默认: $FEATURE_DB)"
    echo -e "  --model_path <路径>       预训练模型路径 (默认: $MODEL_PATH)"
    echo -e "  --batch_size <数量>       批大小 (默认: $BATCH_SIZE)"
    echo -e "  --num_workers <数量>      数据加载器工作进程数 (默认: $NUM_WORKERS)"
    echo -e "  --verbose                 详细输出模式"
    echo -e "  --image_size <尺寸>       输入图像尺寸 (默认: $IMAGE_SIZE)"
    echo -e "  --feat_dim <尺寸>         特征维度 (默认: $FEAT_DIM)"
    echo -e "  --num_views <数量>        每个物体的视点数量 (默认: $NUM_VIEWS)"
    echo -e "  --num_images_per_view <数量> 每个视点的图像数量 (默认: $NUM_IMAGES_PER_VIEW)"
    echo -e "\n示例:"
    echo -e "  $0 --split test --verbose"
    echo -e "  $0 --batch_size 32 --num_workers 8"
    echo -e "  $0 --model_path /path/to/custom/model.pth --feature_db /path/to/output/features.h5"
    echo -e ""
    exit 1
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --root_dir)
            ROOT_DIR="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --feature_db)
            FEATURE_DB="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --image_size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --feat_dim)
            FEAT_DIM="$2"
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
        *)
            echo -e "${RED}未知选项: $1${NC}"
            show_help
            ;;
    esac
done

# 检查必要的路径
if [ ! -d "$ROOT_DIR" ]; then
    echo -e "${RED}错误: 数据集目录不存在: $ROOT_DIR${NC}"
    exit 1
fi

# 创建特征数据库目录
DB_DIR=$(dirname "$FEATURE_DB")
if [ ! -z "$DB_DIR" ]; then
    mkdir -p "$DB_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 无法创建特征数据库目录: $DB_DIR${NC}"
        exit 1
    fi
fi

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}警告: 模型文件不存在: $MODEL_PATH${NC}"
    echo -e "将使用随机初始化的模型进行特征提取。"
fi

# 显示配置信息
echo -e "\n${GREEN}特征提取配置:${NC}"
echo -e "数据集目录:     $ROOT_DIR"
echo -e "数据集分割:     $SPLIT"
echo -e "特征数据库路径: $FEATURE_DB"
echo -e "模型路径:       $MODEL_PATH"
echo -e "批大小:         $BATCH_SIZE"
echo -e "工作进程数:     $NUM_WORKERS"
echo -e "图像尺寸:       $IMAGE_SIZE"
echo -e "特征维度:       $FEAT_DIM"
echo -e "视点数量:       $NUM_VIEWS"
echo -e "每视点图像数:   $NUM_IMAGES_PER_VIEW"
echo -e "详细输出:       $VERBOSE"

# 构建命令
CMD="python extract.py \
    --root_dir $ROOT_DIR \
    --split $SPLIT \
    --feature_db $FEATURE_DB \
    --model_path $MODEL_PATH \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --image_size $IMAGE_SIZE \
    --feat_dim $FEAT_DIM \
    --num_views $NUM_VIEWS \
    --num_images_per_view $NUM_IMAGES_PER_VIEW"

# 如果设置了详细模式，添加参数
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# 显示并执行命令
echo -e "\n${GREEN}执行命令:${NC}"
echo "$CMD"
echo -e "\n${YELLOW}开始特征提取...${NC}"

# 执行特征提取
$CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}特征提取完成！${NC}"
    echo -e "特征已存储到: $FEATURE_DB"
else
    echo -e "\n${RED}特征提取失败！${NC}"
    exit 1
fi