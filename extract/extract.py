#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
import h5py
from torchvision import transforms
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # TensorBoard支持

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型和数据加载器
from models.MultiView_Retrieval_Model import MultiViewRetrievalModel
from tools.Multi_Neighbour_View_Dataloader import MultiViewDataset

# 导入特征验证器 - 使用相对导入或直接导入
# 当在extract目录内运行脚本时，使用直接导入
from feature_check import FeatureValidator, create_feature_dict

class FeatureDatabase:
    """
    特征数据库，用于存储和检索特征
    """
    def __init__(self, db_path, mode='a'):
        """
        :param db_path: 数据库文件路径
        :param mode: 打开模式 ('r'只读, 'a'追加, 'w'写入)
        """
        self.db_path = db_path
        self.mode = mode
        self.db = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        """打开数据库"""
        self.db = h5py.File(self.db_path, self.mode)

    def close(self):
        """关闭数据库"""
        if self.db:
            self.db.close()
            self.db = None

    def store_features(self, obj_id, features):
        """
        存储特征到数据库
        :param obj_id: 物体ID
        :param features: 特征字典
        """
        if not self.db:
            self.open()

        # 创建或更新对象组
        if obj_id in self.db:
            obj_group = self.db[obj_id]
        else:
            obj_group = self.db.create_group(obj_id)

        # 存储特征
        for key, value in features.items():
            if key in obj_group:
                del obj_group[key]  # 删除现有数据集
            obj_group.create_dataset(key, data=value.cpu().numpy())

def load_retrieval_model(args, device):
    """
    加载用于检索的模型
    
    Args:
        args: 命令行参数
        device: 计算设备
    
    Returns:
        model: 加载好的模型
    """
    print("加载检索模型...")

    # 获取特征激活缩放系数（与training模块保持一致）
    feat_activation_scaling = getattr(args, 'feat_activation_scaling', 1.0)
    print(f"特征激活缩放系数: {feat_activation_scaling}")

    # 初始化模型
    model = MultiViewRetrievalModel(
        num_classes=1,  # 占位值，检索时不重要
        feat_dim=args.feat_dim,
        feat_activation_scaling=feat_activation_scaling
    )

    model_path = args.model_path

    # 如果未提供模型路径，尝试使用默认路径
    if not model_path:
        default_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(default_model_path):
            model_path = default_model_path
            print(f"使用默认模型路径: {model_path}")

    # 加载预训练权重
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # 处理DataParallel保存的权重
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        print(f"已加载模型权重: {model_path}")
    else:
        print("警告: 未找到模型文件，使用随机初始化模型")

    # 注意：不在这里调用model.to(device)，而是让调用函数决定何时以及如何移动模型到设备
    return model

def extract_features(args, model=None, tensorboard_dir='../tensorboard_log/validation'):
    """
    提取特征并保存到特征数据库
    
    Args:
        args: 命令行参数
        model: 可选，预加载的模型
        tensorboard_dir: TensorBoard日志保存目录
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到{num_gpus}块可用GPU")

    # 确保特征数据库目录存在
    db_dir = os.path.dirname(args.feature_db)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    # 创建TensorBoard记录器
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    log_step = 0
    
    print(f"TensorBoard日志将保存至: {tensorboard_dir}")
    
    # 加载模型（如果没有提供）
    if model is None:
        model = load_retrieval_model(args, device)
    try:
        # 获取模型第一个可训练参数
        first_param = next(model.parameters())
        # 记录到TensorBoard而不是控制台输出
        writer.add_scalar('Model/param_mean', first_param.mean().item(), 0)
        writer.add_scalar('Model/param_std', first_param.std().item(), 0)
        print(f"模型参数统计已记录到TensorBoard")
    except StopIteration:
        print("警告：模型没有可训练参数（可能未正确初始化）")
        writer.add_text('Warnings/no_trainable_params', '模型没有可训练参数（可能未正确初始化）', 0)

    # 多GPU支持 - 正确的顺序：先确保模型在主设备上，再包装DataParallel
    if num_gpus > 1:
        print(f"使用 {num_gpus} 块GPU进行并行处理")
        # DataParallel要求模型先在device_ids[0]上
        device_ids = list(range(num_gpus))
        model = model.to(device_ids[0])  # 确保模型在第一个GPU上
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        # 单GPU情况
        model = model.to(device)

    # 设置模型为评估模式
    model.eval()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset = MultiViewDataset(
        root_dir=args.root_dir,
        transform=transform,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view,
        split=args.split  # 可以指定提取的数据集分割
    )

    print(f"数据集包含 {len(dataset)} 个样本")

    # 初始化特征验证器
    validator = FeatureValidator()

    # 提取并存储特征
    with FeatureDatabase(args.feature_db, mode='w') as db:
        print("=" * 60)
        print(f"开始提取特征并存储到数据库: {args.feature_db}")
        print("=" * 60)

        # 使用DataLoader批量处理
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 初始化特征验证器
        feature_validator = FeatureValidator()
        print(f"特征验证器已初始化")

        # 用于tqdm的自定义后缀函数
        def get_postfix_dict():
            return {
                '验证': feature_validator.get_batch_summary(batch_results) if 'batch_results' in locals() else '初始化中...',
                '日志步数': log_step
            }

        # 使用tqdm创建进度条
        with tqdm(total=len(dataloader), desc="特征提取") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # 获取数据
                obj_ids = batch['obj_id']
                images = batch['images']  # [batch_size, num_views, num_images_per_view, C, H, W]

                # 显示当前batch的信息
                if args.verbose:
                    print(f"\nBatch {batch_idx}: 包含{len(obj_ids)}个样本")
                    print(f" Batch样本ID: {obj_ids}")

                # 分离三个视点组并移动到设备
                view1 = images[:, 0].to(device, non_blocking=True)  # [batch_size, num_images_per_view, C, H, W]
                view2 = images[:, 1].to(device, non_blocking=True)
                view3 = images[:, 2].to(device, non_blocking=True)

                # 前向传播
                if args.verbose:
                    print(f"  正在处理Batch {batch_idx}的特征提取...")
                with torch.no_grad():
                    outputs = model(view1, view2, view3)

                # 提取特征
                view_features = outputs['view_feats']
                mid_features = outputs['mid_feat']
                global_features = outputs['global_feat']
                obj_feats = outputs['obj_feat']
                
                # 记录批次特征统计到TensorBoard
                writer.add_scalar('Features_batch/view_feats_mean', view_features.mean().item(), log_step)
                writer.add_scalar('Features_batch/view_feats_std', view_features.std().item(), log_step)
                writer.add_scalar('Features_batch/mid_feat_mean', mid_features.mean().item(), log_step)
                writer.add_scalar('Features_batch/mid_feat_std', mid_features.std().item(), log_step)
                writer.add_scalar('Features_batch/global_feat_mean', global_features.mean().item(), log_step)
                writer.add_scalar('Features_batch/global_feat_std', global_features.std().item(), log_step)
                writer.add_scalar('Features_batch/obj_feat_mean', obj_feats.mean().item(), log_step)
                writer.add_scalar('Features_batch/obj_feat_std', obj_feats.std().item(), log_step)

                # 准备批次特征用于验证
                batch_features = []
                for i in range(len(obj_ids)):
                    features = create_feature_dict(
                        view_features[i],
                        mid_features[i].unsqueeze(0),
                        global_features[i].unsqueeze(0),
                        obj_feats[i]
                    )
                    batch_features.append(features)
                
                # 验证特征
                batch_results = validator.validate_batch(batch_features, obj_ids)
                writer.add_scalar('Validation/samples_with_warnings', batch_results['samples_with_warnings'], log_step)
                writer.add_scalar('Validation/total_samples', batch_results['total_samples'], log_step)
                
                # 更新进度条，显示验证摘要
                pbar.set_postfix(get_postfix_dict())
                pbar.update(1)

                # 如果有异常样本，显示警告
                if batch_results['samples_with_warnings'] > 0 and args.verbose:
                    print(f"  ⚠️  警告: 该批次中有 {batch_results['samples_with_warnings']} 个样本特征存在异常")
                    writer.add_text('Warnings/batch_warnings', f'Batch {batch_idx}: 有 {batch_results["samples_with_warnings"]} 个样本特征存在异常', log_step)

                # 存储特征并记录单个样本的特征统计
                for i, obj_id in enumerate(obj_ids):
                    features_to_store = {
                        'view_features': view_features[i].cpu(),
                        'mid_features': mid_features[i].cpu().unsqueeze(0),
                        'global_features': global_features[i].cpu().unsqueeze(0),
                        'obj_feat': obj_feats[i].cpu(),
                    }
                    
                    # 记录单个对象的特征统计
                    writer.add_scalar(f'Features_single/{obj_id}_view_feats_norm', torch.norm(view_features[i]).item(), log_step)
                    writer.add_scalar(f'Features_single/{obj_id}_global_feat_norm', torch.norm(global_features[i]).item(), log_step)
                    writer.add_scalar(f'Features_single/{obj_id}_obj_feat_norm', torch.norm(obj_feats[i]).item(), log_step)
                    
                    # 每100个对象记录一次特征分布直方图
                    if log_step % 100 == 0:
                        writer.add_histogram('Features_dist/global_feat', global_features[i].cpu().numpy(), log_step // 100)
                    
                    db.store_features(obj_id, features_to_store)
                    log_step += 1

    # 打印特征验证整体摘要
    validator.print_overall_summary()
    
    # 关闭TensorBoard记录器
    writer.close()
    
    print(f"\n特征提取完成，存储到: {args.feature_db}")
    print(f"总共处理了 {validator.stats['total_samples']} 个样本")
    print(f"TensorBoard日志已保存至: {tensorboard_dir}")

def parse_args():
    """
    解析命令行参数
    
    Returns:
        args: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='多视图检索系统 - 特征提取模块')

    # 数据参数
    parser.add_argument('--root_dir', type=str, default='/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0',
                        help='数据集根目录')
    parser.add_argument('--split', type=str, default='train',
                        help='数据集分割 (train/val/test)')
    parser.add_argument('--num_views', type=int, default=3,
                        help='每个物体的视点数量')
    parser.add_argument('--num_images_per_view', type=int, default=5,
                        help='每个视点的图像数量')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器的工作进程数')

    # 模型参数
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度大小')
    parser.add_argument('--feat_activation_scaling', type=float, default=1.0,
                        help='特征激活缩放系数（与training模块保持一致）')
    parser.add_argument('--model_path', type=str, default='',
                        help='预训练模型路径')
    parser.add_argument('--save_dir', type=str, default='../checkpoints',
                        help='模型保存目录（用于查找默认模型）')

    # 特征提取参数
    parser.add_argument('--feature_db', type=str, default='../features/feature_db.h5',
                        help='特征数据库路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出模式')

    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 打印参数设置
    print("=" * 50)
    print("参数配置:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("=" * 50)
    
    # 检查路径
    if not args.model_path:
        # 尝试查找默认模型
        default_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(default_model_path):
            print(f"使用默认模型: {default_model_path}")
            args.model_path = default_model_path
        else:
            print("警告: 未找到默认模型，将使用随机初始化的模型")
    
    # 执行特征提取
    extract_features(args)

if __name__ == "__main__":
    main()