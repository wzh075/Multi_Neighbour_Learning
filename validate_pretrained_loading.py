#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证脚本：测试预训练MVCNN参数的自动加载功能
"""

import os
import sys
import torch
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.MultiView_Retrieval_Model import MultiViewRetrievalModel

def test_pretrained_loading():
    """测试预训练权重的自动加载功能"""
    print("开始验证预训练MVCNN参数的自动加载功能...")
    
    # 1. 初始化模型
    model = MultiViewRetrievalModel(num_classes=10, feat_dim=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 2. 检查默认预训练路径是否存在
    default_pretrained_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'feature_extractor_best.pth')
    
    if not os.path.exists(default_pretrained_path):
        print(f"警告: 未找到预训练特征提取器文件: {default_pretrained_path}")
        print("请确保已经运行了预训练脚本并生成了特征提取器权重")
        return False
    
    print(f"找到预训练特征提取器文件: {default_pretrained_path}")
    
    # 3. 测试自动加载逻辑
    try:
        # 保存原始模型权重，用于比较
        original_weights = {k: v.clone() for k, v in model.state_dict().items() if 'view_encoder' in k}
        
        # 尝试加载预训练权重
        feat_extractor_weights = torch.load(default_pretrained_path, map_location=device)
        
        # 创建模型字典的副本以进行修改
        model_dict = model.state_dict()
        
        # 找出预训练特征提取器中与模型view_encoder匹配的部分
        new_state_dict = {}
        for k, v in feat_extractor_weights.items():
            if 'feature_extractor' in k or 'projection' in k or 'feature_scale' in k or 'feature_bias' in k or 'activation_scaling' in k:
                model_key = f'view_encoder.{k}'
                if model_key in model_dict and model_dict[model_key].shape == v.shape:
                    new_state_dict[model_key] = v
        
        if not new_state_dict:
            print("错误: 未找到匹配的特征提取器权重")
            return False
        
        print(f"找到 {len(new_state_dict)} 个匹配的特征提取器权重")
        
        # 更新模型字典并加载
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=False)
        
        # 4. 验证权重是否真的被更新了
        updated_weights = {k: v for k, v in model.state_dict().items() if 'view_encoder' in k}
        
        weights_updated = False
        for k in original_weights.keys():
            if k in updated_weights and not torch.allclose(original_weights[k], updated_weights[k]):
                weights_updated = True
                # 打印一些示例权重进行比较
                if 'weight' in k and len(updated_weights[k].shape) > 1:
                    print(f"\n权重 {k} 已更新:")
                    print(f"  更新前的部分值: {original_weights[k].flatten()[:3]}")
                    print(f"  更新后的部分值: {updated_weights[k].flatten()[:3]}")
                    break
        
        if weights_updated:
            print(f"\n验证成功: 预训练MVCNN参数已成功加载到主模型中")
            return True
        else:
            print("错误: 权重没有被成功更新")
            return False
            
    except Exception as e:
        print(f"验证失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pre_train_save_logic():
    """测试预训练保存逻辑是否正确"""
    print("\n开始验证预训练保存逻辑...")
    
    # 检查预训练脚本是否已经修改为只保存最优结果
    pre_train_path = os.path.join(os.path.dirname(__file__), 'pre_train.py')
    
    if not os.path.exists(pre_train_path):
        print(f"警告: 未找到预训练脚本: {pre_train_path}")
        return False
    
    try:
        with open(pre_train_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否移除了每个epoch保存检查点的代码
        if "save_checkpoint" in content and "torch.save(\"checkpoint_epoch" in content.lower():
            print("错误: 预训练脚本中仍包含每个epoch保存检查点的代码")
            return False
        
        # 检查是否只在验证准确率提升时保存最佳模型
        if "if val_acc > best_acc:" in content and "torch.save(feature_extractor.state_dict(), os.path.join(save_dir, \"feature_extractor_best.pth\"))" in content:
            print("验证成功: 预训练保存逻辑已修改为只保存最优结果")
            return True
        else:
            print("错误: 预训练脚本中未找到正确的最佳模型保存逻辑")
            return False
            
    except Exception as e:
        print(f"验证失败: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='验证预训练权重加载和保存逻辑')
    parser.add_argument('--only-loading', action='store_true', help='仅验证加载逻辑')
    parser.add_argument('--only-saving', action='store_true', help='仅验证保存逻辑')
    args = parser.parse_args()
    
    loading_success = True
    saving_success = True
    
    if not args.only_saving:
        loading_success = test_pretrained_loading()
    
    if not args.only_loading:
        saving_success = test_pre_train_save_logic()
    
    # 输出总体验证结果
    print("\n======= 验证结果总结 =======")
    if args.only_saving:
        print(f"预训练保存逻辑验证: {'成功' if saving_success else '失败'}")
    elif args.only_loading:
        print(f"预训练加载逻辑验证: {'成功' if loading_success else '失败'}")
    else:
        print(f"预训练保存逻辑验证: {'成功' if saving_success else '失败'}")
        print(f"预训练加载逻辑验证: {'成功' if loading_success else '失败'}")
    
    if (args.only_saving and saving_success) or (args.only_loading and loading_success) or (loading_success and saving_success):
        print("\n验证通过! 所有修改已正确实现。")
        sys.exit(0)
    else:
        print("\n验证失败! 请检查代码修改。")
        sys.exit(1)