#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalConsistencyLoss(nn.Module):
    """
    全局一致性损失函数模块
    
    该损失函数用于确保全局特征(global_feat)能够学习到所有视图特征(view_feats)的共同信息，
    通过最小化全局特征与视图共同特征之间的差异来实现
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, view_feats, global_feat):
        """
        计算全局一致性损失
        
        Args:
            view_feats: 视点特征，形状为(batch, num_views, feat_dim)
            global_feat: 全局特征，形状为(batch, feat_dim)
        
        Returns:
            全局一致性损失值（1-平均余弦相似度）
        """
        batch_size, num_views, feat_dim = view_feats.shape
        
        # 计算所有视图的共同成分（通过均值获得）
        view_common_feat = view_feats.mean(dim=1)  # (batch, feat_dim)
        
        # 移除特征归一化，直接使用原始特征
        # view_common_norm = F.normalize(view_common_feat, p=2, dim=-1)  # 已移除
        # global_norm = F.normalize(global_feat, p=2, dim=-1)  # 已移除
        
        # 计算原始共同特征和全局特征之间的余弦相似度
        cos_sim = F.cosine_similarity(view_common_feat, global_feat, dim=-1)
        
        # 返回1-平均余弦相似度作为损失（余弦相似度越高，损失越小）
        return (1 - cos_sim).mean()  # 损失范围[0,2]
    
    def to(self, device=None, dtype=None):
        """
        将模块移动到指定设备和数据类型
        
        Args:
            device: 目标设备
            dtype: 目标数据类型
        
        Returns:
            更新后的模块实例
        """
        super().to(device=device, dtype=dtype)
        return self