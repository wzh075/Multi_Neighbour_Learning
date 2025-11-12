#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.MultiView_Retrieval_Model import MultiViewRetrievalModel, MultiViewEncoder
from pre_training.pre_train_model import PreMVCNN

def transfer_pretrained_weights(pre_trained_path, output_path):
    """
    将预训练的MVCNN权重迁移到主模型中
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载预训练模型
    print(f"Loading pre-trained model from {pre_trained_path}")
    
    # 检查是否是完整模型还是仅特征提取器
    checkpoint = torch.load(pre_trained_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        # 完整模型检查点
        print("Loading full model checkpoint")
        # 创建临时预训练模型用于加载权重
        temp_pre_model = PreMVCNN(num_classes=1, feat_dim=512)  # 类别数不重要
        temp_pre_model.load_state_dict(checkpoint['model_state_dict'])
        pre_model = temp_pre_model
    else:
        # 仅特征提取器权重
        print("Loading feature extractor weights")
        # 创建特征提取器并加载权重
        pre_model = PreMVCNN(num_classes=1, feat_dim=512)
        
        # 尝试直接加载到特征提取器部分
        try:
            pre_model.feature_extractor.load_state_dict(checkpoint)
            print("Loaded weights to feature extractor")
        except:
            # 如果失败，尝试作为完整模型加载
            pre_model.load_state_dict(checkpoint)
            print("Loaded weights as full model")
    
    # 创建主模型的MultiViewEncoder
    print("Creating target MultiViewEncoder")
    target_encoder = MultiViewEncoder(feat_dim=512)
    
    # 迁移权重
    print("Transferring weights from pre-trained model to target encoder")
    
    # 迁移特征提取器权重 - 使用命名参数匹配而不是简单的顺序匹配
    pre_state_dict = {}
    # 收集预训练模型的特征提取器权重，去除可能的模块前缀
    for name, param in pre_model.feature_extractor.named_parameters():
        # 去除可能的前缀
        clean_name = name
        pre_state_dict[clean_name] = param.data
    
    # 收集目标模型的特征提取器权重
    target_state_dict = {}
    for name, param in target_encoder.feature_extractor.named_parameters():
        # 去除可能的前缀
        clean_name = name
        target_state_dict[clean_name] = param
    
    # 按名称匹配并迁移权重
    transferred_count = 0
    for name, param in target_state_dict.items():
        # 尝试直接名称匹配
        if name in pre_state_dict and pre_state_dict[name].size() == param.size():
            param.data.copy_(pre_state_dict[name])
            print(f"Transferred weight: {name}, size: {param.size()}")
            transferred_count += 1
        else:
            # 尝试部分名称匹配
            found = False
            for pre_name, pre_param in pre_state_dict.items():
                # 检查是否包含相同的层名称（如conv1, bn1, layer1等）
                if (any(layer in name and layer in pre_name for layer in ['conv', 'bn', 'layer', 'downsample']) 
                    and pre_param.size() == param.size()):
                    param.data.copy_(pre_param)
                    print(f"Transferred weight with partial match: target={name}, source={pre_name}, size: {param.size()}")
                    transferred_count += 1
                    found = True
                    break
            if not found:
                print(f"Warning: Could not find matching weight for {name}, size: {param.size()}")
    
    print(f"Successfully transferred {transferred_count} weights to feature extractor")
    
    # 迁移投影层权重 - 使用更安全的命名匹配方式
    pre_proj_state = {}
    for name, param in pre_model.projection.named_parameters():
        pre_proj_state[name] = param.data
    
    target_proj_state = {}
    for name, param in target_encoder.projection.named_parameters():
        target_proj_state[name] = param
    
    proj_transferred = 0
    for name, param in target_proj_state.items():
        if name in pre_proj_state and pre_proj_state[name].size() == param.size():
            param.data.copy_(pre_proj_state[name])
            print(f"Transferred projection weight: {name}, size: {param.size()}")
            proj_transferred += 1
        else:
            # 尝试按层类型匹配（如linear层）
            found = False
            for pre_name, pre_param in pre_proj_state.items():
                if pre_param.size() == param.size():
                    param.data.copy_(pre_param)
                    print(f"Transferred projection weight by size: target={name}, source={pre_name}, size: {param.size()}")
                    proj_transferred += 1
                    found = True
                    break
            if not found:
                print(f"Warning: Could not find matching projection weight for {name}, size: {param.size()}")
    
    print(f"Successfully transferred {proj_transferred} projection weights")
    
    # 迁移特征缩放和偏置参数
    if hasattr(pre_model, 'feature_scale') and hasattr(target_encoder, 'feature_scale'):
        if pre_model.feature_scale.size() == target_encoder.feature_scale.size():
            target_encoder.feature_scale.data.copy_(pre_model.feature_scale.data)
            print("Transferred feature_scale parameter")
    
    if hasattr(pre_model, 'feature_bias') and hasattr(target_encoder, 'feature_bias'):
        if pre_model.feature_bias.size() == target_encoder.feature_bias.size():
            target_encoder.feature_bias.data.copy_(pre_model.feature_bias.data)
            print("Transferred feature_bias parameter")
    
    # 保存迁移后的权重
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(target_encoder.state_dict(), output_path)
    print(f"Transferred weights saved to {output_path}")
    
    # 可选：创建主模型并加载迁移后的权重进行测试
    print("\nCreating main retrieval model for testing")
    main_model = MultiViewRetrievalModel(num_classes=1, feat_dim=512)
    
    # 加载迁移后的权重到主模型的view_encoder
    main_model.view_encoder.load_state_dict(torch.load(output_path, map_location=device))
    print("Successfully loaded transferred weights into main model's view_encoder")
    
    # 保存完整的主模型（可选）
    full_model_path = output_path.replace('.pth', '_full_model.pth')
    torch.save(main_model.state_dict(), full_model_path)
    print(f"Full main model with transferred weights saved to {full_model_path}")
    
    return output_path

def create_pretrained_main_model(pre_trained_path, output_model_path, num_classes=1, feat_dim=512):
    """
    创建一个预加载了迁移权重的完整主模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建主模型
    main_model = MultiViewRetrievalModel(num_classes=num_classes, feat_dim=feat_dim)
    
    # 先迁移权重到view_encoder
    temp_encoder_path = output_model_path.replace('.pth', '_encoder.pth')
    transfer_pretrained_weights(pre_trained_path, temp_encoder_path)
    
    # 加载迁移后的权重到主模型
    main_model.view_encoder.load_state_dict(torch.load(temp_encoder_path, map_location=device))
    
    # 保存完整模型
    torch.save({
        'model_state_dict': main_model.state_dict(),
        'pretrained_from': pre_trained_path,
        'feat_dim': feat_dim
    }, output_model_path)
    
    print(f"Created main model with pretrained weights: {output_model_path}")
    return output_model_path

def main():
    parser = argparse.ArgumentParser(description='迁移预训练权重到主模型')
    
    parser.add_argument('--pre_trained_path', type=str, required=True,
                        help='预训练模型权重路径')
    parser.add_argument('--output_path', type=str, default='../pre_checkpoints/transferred_encoder.pth',
                        help='迁移后的权重保存路径')
    parser.add_argument('--create_full_model', action='store_true',
                        help='是否创建完整的主模型')
    parser.add_argument('--full_model_output', type=str, default='../pre_checkpoints/pretrained_main_model.pth',
                        help='完整主模型保存路径')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度')
    
    args = parser.parse_args()
    
    if args.create_full_model:
        create_pretrained_main_model(
            args.pre_trained_path,
            args.full_model_output,
            feat_dim=args.feat_dim
        )
    else:
        transfer_pretrained_weights(args.pre_trained_path, args.output_path)

if __name__ == "__main__":
    main()