#!/bin/bash

# 特征空间可视化脚本
# 这个脚本提供了一个简单的接口来运行特征可视化工具

# 脚本颜色设置
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
FEATURE_DB="../features/feature_db.h5"
OUTPUT_DIR="../visualizations"
METHOD="all"
N_COMPONENTS=2
MAX_OBJECTS_PER_CLASS=50
FEAT_TYPE="obj_feat"
SHOW_FLAG=false

# 显示帮助信息
show_help() {
    echo -e "\n${BLUE}特征空间可视化工具使用说明${NC}\n"
    echo -e "用法: $0 [选项]\n"
    echo -e "选项:"
    echo -e "  -h, --help               显示帮助信息"
    echo -e "  -d, --feature_db PATH    特征数据库路径 (默认: ${FEATURE_DB})"
    echo -e "  -o, --output_dir PATH    输出目录 (默认: ${OUTPUT_DIR})"
    echo -e "  -m, --method METHOD      降维方法 (tsne, pca, umap, all) (默认: ${METHOD})"
    echo -e "  -c, --components NUM     降维维度 (2, 3) (默认: ${N_COMPONENTS})"
    echo -e "  -n, --num_objects NUM    每个类别最多对象数 (默认: ${MAX_OBJECTS_PER_CLASS})"
    echo -e "  -t, --feat_type TYPE     特征类型 (obj_feat, global_features, view_features) (默认: ${FEAT_TYPE})"
    echo -e "  -s, --show               显示图像而不保存\n"
    echo -e "示例:"
    echo -e "  $0                        使用默认参数运行所有可视化"
    echo -e "  $0 -m tsne -c 2           使用t-SNE生成2D可视化"
    echo -e "  $0 -t global_features     可视化全局特征"
    echo -e "  $0 -s                     显示图像而不保存"
    echo -e "  $0 -d ../custom_features.h5 -o ../my_visuals  使用自定义数据库和输出目录\n"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--feature_db)
            FEATURE_DB="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -c|--components)
            N_COMPONENTS="$2"
            shift 2
            ;;
        -n|--num_objects)
            MAX_OBJECTS_PER_CLASS="$2"
            shift 2
            ;;
        -t|--feat_type)
            FEAT_TYPE="$2"
            shift 2
            ;;
        -s|--show)
            SHOW_FLAG=true
            shift
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查特征数据库是否存在
if [ ! -f "$FEATURE_DB" ]; then
    echo -e "${RED}错误: 特征数据库文件不存在: $FEATURE_DB${NC}"
    echo -e "请先运行特征提取脚本生成特征数据库。"
    exit 1
fi

# 创建输出目录
if [ "$SHOW_FLAG" = false ]; then
    mkdir -p "$OUTPUT_DIR"
    echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
fi

# 构建Python命令
PYTHON_CMD="python feature_visualization.py"
PYTHON_CMD+=" --feature_db \"$FEATURE_DB\""
PYTHON_CMD+=" --method \"$METHOD\""
PYTHON_CMD+=" --n_components $N_COMPONENTS"
PYTHON_CMD+=" --max_objects_per_class $MAX_OBJECTS_PER_CLASS"
PYTHON_CMD+=" --feat_type \"$FEAT_TYPE\""

if [ "$SHOW_FLAG" = true ]; then
    PYTHON_CMD+=" --show"
else
    PYTHON_CMD+=" --output_dir \"$OUTPUT_DIR\""
fi

# 显示执行信息
echo -e "\n${BLUE}执行参数:${NC}"
echo -e "  特征数据库: $FEATURE_DB"
echo -e "  输出目录:   $OUTPUT_DIR"
echo -e "  降维方法:   $METHOD"
echo -e "  维度:       $N_COMPONENTS"
echo -e "  每类对象数: $MAX_OBJECTS_PER_CLASS"
echo -e "  特征类型:   $FEAT_TYPE"
echo -e "  显示模式:   ${SHOW_FLAG}"

# 执行可视化命令
echo -e "\n${YELLOW}开始执行可视化...${NC}\n"
if [ "$SHOW_FLAG" = true ]; then
    python feature_visualization.py \
        --feature_db "$FEATURE_DB" \
        --method "$METHOD" \
        --n_components $N_COMPONENTS \
        --max_objects_per_class $MAX_OBJECTS_PER_CLASS \
        --feat_type "$FEAT_TYPE" \
        --show
else
    python feature_visualization.py \
        --feature_db "$FEATURE_DB" \
        --method "$METHOD" \
        --n_components $N_COMPONENTS \
        --max_objects_per_class $MAX_OBJECTS_PER_CLASS \
        --feat_type "$FEAT_TYPE" \
        --output_dir "$OUTPUT_DIR"
fi

# 检查执行结果
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}可视化执行成功!${NC}"
    if [ "$SHOW_FLAG" = false ]; then
        echo -e "  结果保存在: $OUTPUT_DIR"
    fi
else
    echo -e "\n${RED}可视化执行失败!${NC}"
    exit 1
fi