#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loss_function.InfoNCE_Loss import InfoNCELoss
from loss_function.view_similarity_Loss import ViewSimilarityLoss
from loss_function.global_consistency_Loss import GlobalConsistencyLoss

def test_loss_functions():
    """测试调整后的损失函数"""
    print("=== 测试去除归一化后的损失函数 ===")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建损失函数实例
    infonce_loss_fn = InfoNCELoss(tau=0.1)
    view_sim_loss_fn = ViewSimilarityLoss()
    global_consistency_loss_fn = GlobalConsistencyLoss()
    
    # 模拟未归一化的特征数据（特征值范围更大）
    batch_size = 4
    num_views = 3
    feat_dim = 512
    
    # 生成模拟数据 - 未归一化特征（值范围更大）
    torch.manual_seed(42)
    view_feats = torch.randn(batch_size, num_views, feat_dim) * 2.0 + 0.5  # 均值0.5，标准差2.0
    global_feat = torch.randn(batch_size, feat_dim) * 2.0 + 0.5
    obj_ids = [f"obj_{i}" for i in range(batch_size)]
    
    print(f"视图特征形状: {view_feats.shape}")
    print(f"视图特征统计 - 均值: {view_feats.mean():.4f}, 标准差: {view_feats.std():.4f}")
    print(f"全局特征统计 - 均值: {global_feat.mean():.4f}, 标准差: {global_feat.std():.4f}")
    
    # 测试InfoNCE损失
    print("\n--- 测试InfoNCE损失 ---")
    infonce_loss = infonce_loss_fn(view_feats, obj_ids)
    print(f"InfoNCE损失: {infonce_loss.item():.6f}")
    
    # 测试视图相似度损失
    print("\n--- 测试视图相似度损失 ---")
    view_sim_loss = view_sim_loss_fn(view_feats)
    print(f"视图相似度损失: {view_sim_loss.item():.6f}")
    
    # 测试全局一致性损失
    print("\n--- 测试全局一致性损失 ---")
    global_consistency_loss = global_consistency_loss_fn(view_feats, global_feat)
    print(f"全局一致性损失: {global_consistency_loss.item():.6f}")
    
    # 测试不同特征尺度下的损失行为
    print("\n--- 测试不同特征尺度 ---")
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    for scale in scales:
        scaled_view_feats = view_feats * scale
        scaled_global_feat = global_feat * scale
        
        infonce_loss_scaled = infonce_loss_fn(scaled_view_feats, obj_ids)
        view_sim_loss_scaled = view_sim_loss_fn(scaled_view_feats)
        global_consistency_loss_scaled = global_consistency_loss_fn(scaled_view_feats, scaled_global_feat)
        
        print(f"尺度 {scale:4.1f}: InfoNCE={infonce_loss_scaled.item():.6f}, "
              f"视图相似度={view_sim_loss_scaled.item():.6f}, "
              f"全局一致性={global_consistency_loss_scaled.item():.6f}")
    
    # 测试梯度计算
    print("\n--- 测试梯度计算 ---")
    view_feats.requires_grad_(True)
    global_feat.requires_grad_(True)
    
    total_loss = infonce_loss_fn(view_feats, obj_ids) + \
                 view_sim_loss_fn(view_feats) + \
                 global_consistency_loss_fn(view_feats, global_feat)
    
    total_loss.backward()
    
    print(f"总损失: {total_loss.item():.6f}")
    print(f"视图特征梯度范数: {view_feats.grad.norm().item():.6f}")
    print(f"全局特征梯度范数: {global_feat.grad.norm().item():.6f}")
    
    # 检查梯度是否正常（不为零且不过大）
    if view_feats.grad.norm() > 1e-10 and view_feats.grad.norm() < 1e6:
        print("✓ 梯度计算正常")
    else:
        print("✗ 梯度计算异常")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_loss_functions()