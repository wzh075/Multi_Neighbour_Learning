#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
损失函数模块化使用示例脚本

本脚本演示如何单独使用拆分后的三个损失函数模块：
1. InfoNCELoss - 对比学习损失
2. ViewSimilarityLoss - 视图相似度损失
3. GlobalConsistencyLoss - 全局一致性损失

通过本示例，您可以了解每个损失函数的独立使用方法，以及如何根据需求组合它们。
"""

import torch
import torch.nn as nn

# 导入各个损失函数模块
from InfoNCE_Loss import InfoNCELoss
from view_similarity_Loss import ViewSimilarityLoss
from global_consistency_Loss import GlobalConsistencyLoss


# 生成示例数据
def generate_sample_data(batch_size=4, num_views=3, feat_dim=256):
    """
    生成用于演示的样本数据
    
    Args:
        batch_size: 批次大小
        num_views: 每个对象的视图数量
        feat_dim: 特征维度
    
    Returns:
        包含示例数据的字典
    """
    # 随机生成示例特征
    view_feats = torch.randn(batch_size, num_views, feat_dim)
    global_feat = torch.randn(batch_size, feat_dim)
    obj_feat = torch.randn(batch_size, feat_dim)
    
    # 生成对象ID和类别标签
    obj_ids = torch.arange(batch_size)
    # 为简单起见，假设两个类别，每个类别有batch_size/2个样本
    class_labels = torch.cat([torch.zeros(batch_size // 2), torch.ones(batch_size // 2)]).long()
    
    return {
        "view_feats": view_feats,
        "global_feat": global_feat,
        "obj_feat": obj_feat,
        "obj_ids": obj_ids,
        "class_labels": class_labels
    }


def demonstrate_individual_losses(sample_data, device=None):
    """
    演示如何单独使用每个损失函数
    
    Args:
        sample_data: 包含示例数据的字典
        device: 计算设备
    """
    # 将数据移动到指定设备
    if device is not None:
        for key, value in sample_data.items():
            if isinstance(value, torch.Tensor):
                sample_data[key] = value.to(device)
    
    print("\n=== 演示单独使用各个损失函数 ===\n")
    
    # 1. 使用InfoNCELoss
    print("1. InfoNCELoss:")
    infonce_loss = InfoNCELoss(tau=0.5)
    if device is not None:
        infonce_loss = infonce_loss.to(device)
    loss_infonce = infonce_loss(sample_data["view_feats"], sample_data["obj_ids"])
    print(f"   损失值: {loss_infonce.item():.4f}")
    
    # 2. 使用ViewSimilarityLoss
    print("\n2. ViewSimilarityLoss:")
    view_sim_loss = ViewSimilarityLoss()
    if device is not None:
        view_sim_loss = view_sim_loss.to(device)
    loss_view_sim = view_sim_loss(sample_data["view_feats"])
    print(f"   损失值: {loss_view_sim.item():.4f}")
    
    # 3. 使用GlobalConsistencyLoss
    print("\n3. GlobalConsistencyLoss:")
    global_cons_loss = GlobalConsistencyLoss()
    if device is not None:
        global_cons_loss = global_cons_loss.to(device)
    loss_global_cons = global_cons_loss(sample_data["view_feats"], sample_data["global_feat"])
    print(f"   损失值: {loss_global_cons.item():.4f}")
    
    return {
        "infonce_loss": loss_infonce,
        "view_sim_loss": loss_view_sim,
        "global_cons_loss": loss_global_cons
    }


def demonstrate_combined_losses(sample_data, device=None):
    """
    演示如何组合使用多个损失函数
    
    这是一个简单的组合示例，实际使用时可以根据需求灵活组合各个损失函数。
    
    Args:
        sample_data: 包含示例数据的字典
        device: 计算设备
    """
    # 将数据移动到指定设备
    if device is not None:
        for key, value in sample_data.items():
            if isinstance(value, torch.Tensor):
                sample_data[key] = value.to(device)
    
    print("\n=== 演示组合使用损失函数 ===\n")
    
    # 初始化各个损失函数
    infonce_loss = InfoNCELoss(tau=0.5)
    view_sim_loss = ViewSimilarityLoss()
    global_cons_loss = GlobalConsistencyLoss()
    
    # 将损失函数移动到指定设备
    if device is not None:
        infonce_loss = infonce_loss.to(device)
        view_sim_loss = view_sim_loss.to(device)
        global_cons_loss = global_cons_loss.to(device)
    
    # 设置各损失函数的权重
    lambda_view_sim = 0.0001
    lambda_global_consistency = 1.0
    
    # 计算各个损失
    loss_infonce = infonce_loss(sample_data["view_feats"], sample_data["obj_ids"])
    loss_view_sim = lambda_view_sim * view_sim_loss(sample_data["view_feats"])
    loss_global_cons = lambda_global_consistency * global_cons_loss(
        sample_data["view_feats"], sample_data["global_feat"]
    )
    
    # 计算总损失
    total_loss = loss_infonce + loss_view_sim + loss_global_cons
    
    # 打印结果
    print(f"InfoNCE损失: {loss_infonce.item():.4f}")
    print(f"视图相似度损失(权重 {lambda_view_sim}): {loss_view_sim.item():.4f}")
    print(f"全局一致性损失(权重 {lambda_global_consistency}): {loss_global_cons.item():.4f}")
    print(f"总损失: {total_loss.item():.4f}")
    
    return {
        "loss_infonce": loss_infonce,
        "loss_view_sim": loss_view_sim,
        "loss_global_cons": loss_global_cons,
        "total_loss": total_loss
    }


def demonstrate_training_loop(sample_data, device=None):
    """
    演示在训练循环中如何使用这些损失函数
    
    Args:
        sample_data: 包含示例数据的字典
        device: 计算设备
    """
    print("\n=== 演示在训练循环中使用损失函数 ===\n")
    
    # 初始化损失函数
    infonce_loss = InfoNCELoss(tau=0.5)
    view_sim_loss = ViewSimilarityLoss()
    global_cons_loss = GlobalConsistencyLoss()
    
    # 将损失函数移动到指定设备
    if device is not None:
        infonce_loss = infonce_loss.to(device)
        view_sim_loss = view_sim_loss.to(device)
        global_cons_loss = global_cons_loss.to(device)
    
    # 设置各损失函数的权重
    lambda_view_sim = 0.0001
    lambda_global_consistency = 1.0
    
    # 模拟训练循环（3个epoch）
    for epoch in range(1, 4):
        print(f"\nEpoch {epoch}:")
        
        # 模拟模型输出
        # 注意：实际使用时，这些特征应该来自您的模型前向传播
        model_output = {
            "view_feats": sample_data["view_feats"].clone().requires_grad_(True),
            "global_feat": sample_data["global_feat"].clone().requires_grad_(True)
        }
        
        # 计算各个损失
        loss_infonce = infonce_loss(model_output["view_feats"], sample_data["obj_ids"])
        loss_view_sim = lambda_view_sim * view_sim_loss(model_output["view_feats"])
        loss_global_cons = lambda_global_consistency * global_cons_loss(
            model_output["view_feats"], model_output["global_feat"]
        )
        
        # 计算总损失
        total_loss = loss_infonce + loss_view_sim + loss_global_cons
        
        # 打印损失信息
        print(f"  InfoNCE: {loss_infonce.item():.4f}, "
              f"ViewSim: {loss_view_sim.item():.4f}, "
              f"GlobalCons: {loss_global_cons.item():.4f}, "
              f"Total: {total_loss.item():.4f}")
        
        # 模拟反向传播和参数更新
        # 注意：实际使用时，您需要使用optimizer来更新模型参数
        total_loss.backward()
        
        # 打印梯度信息（用于演示）
        print(f"  view_feats梯度范数: {model_output['view_feats'].grad.norm().item():.6f}")
        print(f"  global_feat梯度范数: {model_output['global_feat'].grad.norm().item():.6f}")
        print(f"  obj_feat梯度范数: {model_output['obj_feat'].grad.norm().item():.6f}")


if __name__ == "__main__":
    # 设置设备（如果有GPU则使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成示例数据
    sample_data = generate_sample_data(batch_size=4, num_views=3, feat_dim=256)
    
    # 演示单独使用各个损失函数
    individual_losses = demonstrate_individual_losses(sample_data, device)
    
    # 演示组合使用损失函数
    combined_losses = demonstrate_combined_losses(sample_data, device)
    
    # 演示在训练循环中使用损失函数
    demonstrate_training_loop(sample_data, device)
    
    print("\n=== 示例脚本运行完成 ===\n")
    print("您现在可以根据需要，在自己的训练代码中导入并使用这些损失函数模块。")
    print("例如：")
    print("\nfrom loss_function.InfoNCE_Loss import InfoNCELoss")
    print("from loss_function.view_similarity_Loss import ViewSimilarityLoss")
    print("\n# 初始化损失函数")
    print("infonce_loss = InfoNCELoss(tau=0.5)")
    print("view_sim_loss = ViewSimilarityLoss()")
    print("\n# 在训练循环中使用")
    print("loss_infonce = infonce_loss(view_feats, obj_ids)")
    print("loss_view_sim = lambda_view_sim * view_sim_loss(view_feats)")
    print("\n# 计算总损失")
    print("total_loss = loss_infonce + loss_view_sim")
    print("\n# 反向传播")
    print("total_loss.backward()")