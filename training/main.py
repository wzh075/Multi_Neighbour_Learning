#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
from torchvision import transforms
from torch import nn

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入训练模块
try:
    from training.train import train_model
except ImportError:
    print("错误: 无法导入训练模块，请检查路径")
    sys.exit(1)

# 导入模型、数据加载器和损失函数
try:
    from models.MultiView_Retrieval_Model import MultiViewRetrievalModel
    from tools.Multi_Neighbour_View_Dataloader import MultiViewDataset
    from loss_function.InfoNCE_Loss import InfoNCELoss
    from loss_function.view_similarity_Loss import ViewSimilarityLoss
    from loss_function.global_consistency_Loss import GlobalConsistencyLoss
except ImportError as e:
    print(f"错误: 无法导入必要模块: {e}")
    sys.exit(1)

class MultiViewSplitLoss(nn.Module):
    """
    多视图拆分损失函数，整合各个损失组件
    """
    def __init__(self, tau=0.1, lambda_view_sim=0.001, 
                 lambda_global_consistency=0.3, device=None):
        super(MultiViewSplitLoss, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tau = tau
        
        # 初始化各个损失组件
        self.infonce_loss_fn = InfoNCELoss(tau=tau)
        self.view_sim_loss_fn = ViewSimilarityLoss()
        self.global_consistency_loss_fn = GlobalConsistencyLoss()
        
        # 添加特征正则化层，用于改善特征分布
        self.feat_regularizer = nn.Parameter(torch.tensor(1.0))
        
        # 特征正则化权重参数
        self.feat_reg_weight = 1.0  # 默认权重
        
        # 将损失函数移动到指定设备
        self.infonce_loss_fn = self.infonce_loss_fn.to(self.device)
        self.view_sim_loss_fn = self.view_sim_loss_fn.to(self.device)
        self.global_consistency_loss_fn = self.global_consistency_loss_fn.to(self.device)
        
        # 损失权重
        self.lambda_view_sim = lambda_view_sim
        self.lambda_global_consistency = lambda_global_consistency
        # 特征正则化权重
        self.feat_mean_reg_weight = 0.2
        self.feat_var_reg_weight = 0.1
    
    def forward(self, view_feats, mid_feat, global_feat, obj_ids, obj_feat=None, class_labels=None, raw_mid_feat=None, raw_global_feat=None, raw_obj_feat=None):
        """
        前向传播计算总损失，添加特征分布优化
        
        参数:
        - view_feats: 视图特征 [batch_size, num_views, feat_dim]
        - mid_feat: 中间特征 [batch_size, mid_feat_dim]
        - global_feat: 全局特征 [batch_size, feat_dim]
        - obj_ids: 对象ID [batch_size]
        - obj_feat: 对象特征 [batch_size, feat_dim] (可选)
        - class_labels: 类别标签 [batch_size] (可选)
        - raw_mid_feat: 原始中间特征 [batch_size, mid_feat_dim] (可选)
        - raw_global_feat: 原始全局特征 [batch_size, feat_dim] (可选)
        - raw_obj_feat: 原始对象特征 [batch_size, feat_dim] (可选)
        
        返回:
        - 包含各损失项的字典
        """
        # 使用视图特征的原始值进行InfoNCE计算，保持特征多样性
        # InfoNCE损失函数内部已移除归一化，直接使用原始特征
        # 增加InfoNCE损失的权重，确保它在总损失中占有重要比重
        infonce_loss = self.infonce_loss_fn(view_feats, obj_ids) * 2.0  # 提高权重
        # 添加最小损失下限，防止过快收敛到0
        infonce_loss = torch.max(infonce_loss, torch.tensor(0.1).to(self.device))  # 提高最小损失
        
        # 视图相似度损失 - 减小权重，避免过度正则化
        view_similarity_loss = self.view_sim_loss_fn(view_feats) * (self.lambda_view_sim * 0.5)  # 调整为合理的权重
        
        # 全局一致性损失 - 适当增大权重，确保特征一致性
        global_consistency_loss = self.global_consistency_loss_fn(view_feats, global_feat) * (self.lambda_global_consistency * 1.5)  # 适度增大权重
        
        # 对象聚类损失已移除
        
        # 重新实现特征正则化，确保不会导致特征过快收敛
        feat_mean_reg = 0.0
        feat_var_reg = 0.0
        
        # 1. 视图特征正则化 - 确保特征均值远离零
        if view_feats is not None:
            # 对于视图特征，计算每个维度的均值，并鼓励它们接近0.2（更合理的目标值）
            view_mean_reg = torch.mean(torch.abs(view_feats.mean(dim=0).mean(dim=0) - 0.2))
            # 确保方差足够大，防止特征坍塌
            view_var_reg = torch.mean(torch.max(torch.tensor(0.1).to(self.device) - view_feats.var(dim=0).mean(dim=0), torch.tensor(0.0).to(self.device)))
            feat_mean_reg += (view_mean_reg + view_var_reg) * 0.2  # 减小权重
        
        # 2. 原始中间特征正则化
        if raw_mid_feat is not None:
            mid_mean_reg = torch.mean(torch.abs(raw_mid_feat.mean(dim=0) - 0.15))
            mid_var_reg = torch.mean(torch.max(torch.tensor(0.08).to(self.device) - raw_mid_feat.var(dim=0), torch.tensor(0.0).to(self.device)))
            feat_mean_reg += (mid_mean_reg + mid_var_reg) * 0.15  # 减小权重
        
        # 3. 原始全局特征正则化
        if raw_global_feat is not None:
            global_mean_reg = torch.mean(torch.abs(raw_global_feat.mean(dim=0) - 0.2))
            global_var_reg = torch.mean(torch.max(torch.tensor(0.1).to(self.device) - raw_global_feat.var(dim=0), torch.tensor(0.0).to(self.device)))
            feat_mean_reg += (global_mean_reg + global_var_reg) * 0.15  # 减小权重
        
        # 4. 原始对象特征正则化 - 更温和的正则化
        if raw_obj_feat is not None:
            obj_mean_reg = torch.mean(torch.abs(raw_obj_feat.mean(dim=0) - 0.2))
            obj_var_reg = torch.mean(torch.max(torch.tensor(0.1).to(self.device) - raw_obj_feat.var(dim=0), torch.tensor(0.0).to(self.device)))
            feat_mean_reg += (obj_mean_reg + obj_var_reg) * 0.2  # 减小权重
            
            # 添加额外的防止特征坍塌的正则化
            if raw_obj_feat.std().item() < 0.1:
                feat_mean_reg += 0.1  # 当特征方差过小时增加惩罚
        
        # 总特征正则化损失 - 减小整体权重，避免过度正则化
        total_feat_reg = (feat_mean_reg + feat_var_reg) * 0.5
        
        # 定期记录特征统计信息到TensorBoard（如果writer可用）
        if self.training and torch.rand(1) < 0.05 and hasattr(self, 'writer') and self.writer is not None:
            global_step = self.global_step if hasattr(self, 'global_step') else 0
            self.writer.add_scalar('Features/view_feats_mean', view_feats.mean().item(), global_step)
            self.writer.add_scalar('Features/view_feats_std', view_feats.std().item(), global_step)
            if global_feat is not None:
                self.writer.add_scalar('Features/global_feat_mean', global_feat.mean().item(), global_step)
                self.writer.add_scalar('Features/global_feat_std', global_feat.std().item(), global_step)
            if obj_feat is not None:
                self.writer.add_scalar('Features/obj_feat_mean', obj_feat.mean().item(), global_step)
                self.writer.add_scalar('Features/obj_feat_std', obj_feat.std().item(), global_step)
        
        # 返回损失字典，确保所有损失项都被包含
        return {
            'infonce_loss': infonce_loss,
            'view_similarity_loss': view_similarity_loss,
            'global_consistency_loss': global_consistency_loss,
            'feat_mean_reg': total_feat_reg  # 简化正则化权重
        }

def train_model_func(args):
    """训练模型的功能函数
    
    Args:
        args: 命令行参数对象，包含训练配置
        
    Returns:
        训练完成的模型实例
    """
    # 设置设备 - 更健壮的设备检测
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 显示当前使用的CUDA设备信息
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"使用设备: GPU ({device_name})")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU (未检测到可用GPU)")

    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"检测到{num_gpus}块可用GPU")

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置特征激活缩放参数
    feat_activation_scaling = getattr(args, 'feat_activation_scaling', 1.0)
    print(f"特征激活缩放系数: {feat_activation_scaling}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset = MultiViewDataset(
        root_dir=args.root_dir,
        transform=transform,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view,
        split="train"
    )

    # 初始化模型
    model = MultiViewRetrievalModel(
        num_classes=dataset.num_classes,
        feat_dim=args.feat_dim,
        feat_activation_scaling=feat_activation_scaling
    )
    
    # 尝试自动加载预训练MVCNN参数
    pretrained_loaded = False
    
    # 首先检查是否通过命令行参数提供了预训练权重
    if args.pretrained_weights is not None:
        if os.path.exists(args.pretrained_weights):
            print(f"加载预训练权重: {args.pretrained_weights}")
            try:
                # 加载预训练权重
                checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
                
                # 检查是否是完整的模型权重或只有编码器权重
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # 这是完整模型，使用标准加载逻辑
                    is_feature_extractor_only = False
                else:
                    state_dict = checkpoint
                    # 检查是否是特征提取器权重（通常会有projection或feature_extractor等键）
                    is_feature_extractor_only = any(key.startswith('projection') or 'feature_extractor' in key or 'feature_scale' in key for key in state_dict.keys())
                
                # 处理可能的键名不匹配
                model_dict = model.state_dict()
                
                # 创建新的state_dict，只保留模型中存在的键
                new_state_dict = {}
                
                # 如果是特征提取器权重，需要映射键名
                if is_feature_extractor_only:
                    print("检测到特征提取器权重，进行键名映射")
                    for k, v in state_dict.items():
                        # 将特征提取器权重映射到view_encoder
                        model_key = f'view_encoder.{k}'
                        if model_key in model_dict and model_dict[model_key].shape == v.shape:
                            new_state_dict[model_key] = v
                            print(f"映射权重: {k} -> {model_key}")
                else:
                    # 标准模型权重加载逻辑
                    for k, v in state_dict.items():
                        # 移除可能的.module前缀
                        if k.startswith('module.'):
                            k = k[7:]
                        
                        # 检查键是否存在于模型中且形状匹配
                        if k in model_dict and model_dict[k].shape == v.shape:
                            new_state_dict[k] = v
                        elif 'classifier' in k or 'fc' in k.lower() and 'class' in k.lower():
                            # 跳过分类器层，因为类别数可能不同
                            print(f"跳过分类器层: {k}")
                        else:
                            print(f"未使用的权重: {k}, 形状不匹配")
                
                # 更新模型权重
                model_dict.update(new_state_dict)
                model.load_state_dict(model_dict, strict=False)
                print(f"成功加载 {len(new_state_dict)} 个预训练权重参数")
                pretrained_loaded = True
                
                # 冻结编码器部分（如果需要）
                if args.freeze_encoder:
                    print("冻结编码器部分权重")
                    for name, param in model.named_parameters():
                        if 'encoder' in name or 'backbone' in name:
                            param.requires_grad = False
                
            except Exception as e:
                print(f"加载预训练权重失败: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"警告: 预训练权重文件不存在: {args.pretrained_weights}")
    
    # 如果没有通过命令行参数提供，尝试自动查找预训练MVCNN特征提取器权重
    if not pretrained_loaded:
        # 默认的预训练特征提取器权重路径 - 优先查找pre_training保存的路径和文件名
        default_pretrained_path = os.path.join(os.path.dirname(__file__), '..', 'pre_checkpoints', 'best_feature_extractor.pth')
        if os.path.exists(default_pretrained_path):
            print(f"自动加载MVCNN特征提取器权重: {default_pretrained_path}")
            try:
                # 加载预训练的特征提取器权重
                feat_extractor_weights = torch.load(default_pretrained_path, map_location='cpu')
                
                # 创建模型字典的副本以进行修改
                model_dict = model.state_dict()
                
                # 找出预训练特征提取器中与模型view_encoder匹配的部分
                # 将预训练权重的键名转换为模型中对应的键名
                new_state_dict = {}
                for k, v in feat_extractor_weights.items():
                    # 特征提取器权重名称映射到view_encoder
                    if 'feature_extractor' in k or 'projection' in k or 'feature_scale' in k or 'feature_bias' in k or 'activation_scaling' in k:
                        model_key = f'view_encoder.{k}'
                        if model_key in model_dict and model_dict[model_key].shape == v.shape:
                            new_state_dict[model_key] = v
                
                # 检查是否成功加载了一些权重
                if not new_state_dict:
                    print("警告: 未找到匹配的特征提取器权重")
                else:
                    print(f"成功加载 {len(new_state_dict)} 个匹配的特征提取器权重")
                    # 更新模型字典
                    model_dict.update(new_state_dict)
                    # 加载更新后的模型字典
                    model.load_state_dict(model_dict, strict=False)
                    print("MVCNN特征提取器权重加载成功")
                    pretrained_loaded = True
            except Exception as e:
                print(f"加载MVCNN特征提取器权重失败: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # 如果没有找到预训练权重，使用默认初始化
    if not pretrained_loaded:
        print("未找到预训练权重，使用默认初始化")

    # 禁用调试模式以减少输出
    try:
        model.view_global_fusion.debug = False
    except AttributeError:
        print("警告: 模型可能没有view_global_fusion.debug属性，忽略此设置")

    # 多GPU支持 - 正确的顺序：先确保模型在主设备上，再包装DataParallel
    if num_gpus > 1:
        print(f"使用 {num_gpus} 块GPU进行并行训练")
        # DataParallel要求模型先在device_ids[0]上
        device_ids = list(range(num_gpus))
        model = model.to(device_ids[0])  # 确保模型在第一个GPU上
        model = nn.DataParallel(model, device_ids=device_ids)
        # 对于DataParallel包装的模型，需要通过.module访问原始模型属性
        try:
            model.module.view_global_fusion.debug = False
        except AttributeError:
            print("警告: 模型可能没有view_global_fusion.debug属性，忽略此设置")
    else:
        # 单GPU或CPU情况
        model = model.to(device)
        print(f"模型已移至设备: {device}")

    # 初始化损失函数（使用模块化的损失组件）
    criterion = MultiViewSplitLoss(
        tau=args.tau,
        lambda_view_sim=args.lambda_view_sim,
        lambda_global_consistency=args.lambda_global_consistency,
        device=device
    )
    
    # 设置特征正则化权重
    criterion.feat_reg_weight = getattr(args, 'feat_reg_weight', 1.0)
    print(f"特征正则化权重设置为: {criterion.feat_reg_weight}")

    print("开始训练模型...")
    print("训练集样本数: {}".format(len(dataset)))
    print("类别数: {}".format(dataset.num_classes))
    print("训练参数 - 轮数: {}, 批大小: {}, 学习率: {}".format(args.epochs, args.batch_size, args.lr))

    # 训练模型
    print("=" * 50)
    print("开始训练模型...")
    print("=" * 50)

    trained_model = train_model(
        model,
        dataset,
        criterion,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=device
    )

    return trained_model

def parse_args():
    """解析命令行参数
    
    Returns:
        解析后的命令行参数对象
    """
    parser = argparse.ArgumentParser(description='多视图检索系统 - 训练模块')

    # 数据参数
    parser.add_argument('--root_dir', type=str, default='/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0',
                        help='数据集根目录')
    parser.add_argument('--num_views', type=int, default=3,
                        help='每个物体的视点数量')
    parser.add_argument('--num_images_per_view', type=int, default=5,
                        help='每个视点的图像数量')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像尺寸')

    # 模型参数
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度大小')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='InfoNCE损失的温度参数')
    parser.add_argument('--lambda_view_sim', type=float, default=0.0001,
                        help='视图特征间相似度损失权重')
    parser.add_argument('--lambda_global_consistency', type=float, default=1.0,
                        help='global_feat一致性损失权重')
    parser.add_argument('--feat_activation_scaling', type=float, default=1.0,
                        help='特征激活缩放系数')
    parser.add_argument('--feat_reg_weight', type=float, default=1.0,
                        help='特征分布正则化权重')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='预训练权重文件路径')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='是否冻结编码器部分权重')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')

    return parser.parse_args()

def main():
    """主函数 - 程序入口点"""
    try:
        # 解析参数
        args = parse_args()
        
        # 打印参数设置
        print("=" * 50)
        print("参数配置:")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print("=" * 50)
        
        # 确保保存目录存在
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"模型保存目录: {os.path.abspath(args.save_dir)}")
        
        # 执行训练
        train_model_func(args)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"错误: 程序执行失败 - {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()