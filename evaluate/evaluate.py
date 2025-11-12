import os
import sys
import json
import torch
import numpy as np
import argparse
from datetime import datetime
from retrieval_sys import RetrievalSystem
from torchvision import transforms
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入正确路径的模块
try:
    from tools.Multi_Neighbour_View_Dataloader import MultiViewDataset
    from models.MultiView_Retrieval_Model import MultiViewRetrievalModel
    
    # 定义load_retrieval_model函数用于加载模型
    def load_retrieval_model(model_path, model_name, feature_dim, device):
        """加载检索模型"""
        # 根据ModelNet40数据集设置类别数量为40
        num_classes = 40
        
        # 创建模型配置，确保与MultiViewRetrievalModel构造函数匹配
        model_config = {
            'num_classes': num_classes,
            'feat_dim': feature_dim
        }
        
        # 初始化模型
        model = MultiViewRetrievalModel(**model_config)
        
        # 加载预训练权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # 处理不同格式的checkpoint
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # 使用strict=False参数忽略不匹配的键，解决模型架构变化问题
            try:
                model.load_state_dict(state_dict, strict=False)
                print("模型权重已加载，忽略不匹配的键")
            except Exception as e:
                print(f"加载模型权重时出现错误: {str(e)}")
                # 尝试手动映射权重键
                new_state_dict = {}
                for key, value in state_dict.items():
                    # 尝试直接匹配或简单映射
                    if key in model.state_dict():
                        new_state_dict[key] = value
                    # 对于可能的基础层名称变化，可以添加映射逻辑
                    # 这里不做复杂映射，仅使用存在的键
                if new_state_dict:
                    model.load_state_dict(new_state_dict, strict=False)
                    print(f"部分权重已加载: {len(new_state_dict)}/{len(state_dict)} 个键")
        
        # 将模型移动到指定设备
        model.to(device)
        model.eval()  # 设置为评估模式
        
        return model
except ImportError as e:
        raise ImportError(f"无法导入必要的模块: {str(e)}")

def setup_logger(log_file):
    """设置日志记录器
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def evaluate_retrieval_system(model, dataset, args, logger):
    """评估检索系统性能
    
    Args:
        model: 特征提取模型
        dataset: 评估数据集
        args: 命令行参数
        logger: 日志记录器
    
    Returns:
        评估结果字典
    """
    # 创建TensorBoard目录
    tensorboard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tensorboard_log', 'retrieval')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # 初始化检索系统
    retrieval_system = RetrievalSystem(model=model, tensorboard_dir=tensorboard_dir)
    logger.info(f"TensorBoard日志将保存至: {tensorboard_dir}")
    
    # 构建特征数据库
    logger.info(f"开始构建特征数据库，数据集大小: {len(dataset)}")
    retrieval_system.build_database(dataset, batch_size=args.batch_size, device=args.device)
    
    # 评估对象检索性能
    logger.info("开始评估对象检索性能")
    obj_retrieval_results = retrieval_system.evaluate_object_retrieval(
        dataset, 
        topk_list=args.topk_list,
        use_multiple_views=args.use_multiple_views,
        debug=args.debug
    )
    
    # 记录对象检索结果
    logger.info("对象检索性能评估结果:")
    for mode in ['single_view', 'multiple_views', 'overall']:
        if mode in obj_retrieval_results:
            logger.info(f"\n{mode.upper()} 结果:")
            for topk in args.topk_list:
                if topk in obj_retrieval_results[mode]:
                    result = obj_retrieval_results[mode][topk]
                    logger.info(f"  Top-{topk}:")
                    logger.info(f"    对象召回率: {result['obj_recall']:.4f}")
                    logger.info(f"    类别召回率: {result['class_recall']:.4f}")
                    if 'total_queries' in result:
                        logger.info(f"    查询总数: {result['total_queries']}")
    
    # 评估视图检索性能
    logger.info("\n开始评估视图检索性能")
    view_retrieval_results = retrieval_system.evaluate_view_retrieval(
        dataset, 
        topk_list=args.topk_list
    )
    
    # 记录视图检索结果
    logger.info("视图检索性能评估结果:")
    for topk in args.topk_list:
        if topk in view_retrieval_results:
            result = view_retrieval_results[topk]
            logger.info(f"  Top-{topk}:")
            logger.info(f"    视图召回率: {result['view_recall']:.4f}")
            logger.info(f"    查询总数: {result['total_queries']}")
    
    # 整合所有结果
    all_results = {
        'object_retrieval': obj_retrieval_results,
        'view_retrieval': view_retrieval_results,
        'args': vars(args)
    }
    
    # 保存评估结果
    if args.output_dir:
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"评估结果已保存到: {results_file}")
    
    return all_results

def parse_args():
    """解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='多视图检索系统评估')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='数据集划分')
    parser.add_argument('--num_views', type=int, default=3, help='每个对象的视图数量')
    parser.add_argument('--num_images_per_view', type=int, default=5, help='每个视图的图像数量')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--model_name', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], help='模型名称')
    parser.add_argument('--feature_dim', type=int, default=1024, help='特征维度')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='运行设备')
    parser.add_argument('--topk_list', type=int, nargs='+', default=[1, 5, 10], help='评估的top-k值列表')
    parser.add_argument('--use_multiple_views', action='store_true', help='是否使用多视图检索')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='评估结果输出目录')
    parser.add_argument('--log_file', type=str, default=None, help='日志文件路径（默认在output_dir中生成）')
    
    # 其他参数
    parser.add_argument('--debug', action='store_true', help='是否打印调试信息')
    
    args = parser.parse_args()
    
    # 参数验证
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"数据集目录不存在: {args.data_dir}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 如果未指定日志文件路径，则在输出目录中生成
    if args.log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = args.output_dir if args.output_dir else '.'
        args.log_file = os.path.join(log_dir, f'evaluation_{timestamp}.log')
    
    return args

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志记录
    logger = setup_logger(args.log_file)
    logger.info(f"开始评估多视图检索系统")
    logger.info(f"命令行参数: {vars(args)}")
    
    try:
        # 检查CUDA可用性
        if args.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU")
            args.device = 'cpu'
        
        # 加载模型
        logger.info(f"加载模型: {args.model_path}")
        model = load_retrieval_model(args.model_path, args.model_name, args.feature_dim, args.device)
        
        # 数据预处理 - 与训练脚本保持一致的transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 使用与训练相同的图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        logger.info(f"加载数据集: {args.data_dir} (split: {args.split})")
        dataset = MultiViewDataset(
            root_dir=args.data_dir,
            transform=transform,
            num_views=args.num_views,
            num_images_per_view=args.num_images_per_view,
            split=args.split
        )
        
        # 执行评估
        results = evaluate_retrieval_system(model, dataset, args, logger)
        
        logger.info("评估完成！")
        
    except Exception as e:
        logger.error(f"评估过程中出现错误: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()