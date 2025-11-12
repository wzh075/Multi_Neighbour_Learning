#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewSimilarityLoss(nn.Module):
    """
    视图相似度损失函数模块
    
    该损失函数用于确保同一对象的不同视图之间保持一定的相似度，
    通过最小化1-平均余弦相似度来实现
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, view_feats):
        """
        计算视图相似度损失
        
        Args:
            view_feats: 视点特征，形状为(batch, num_views, feat_dim)
        
        Returns:
            视图相似度损失值（1-平均余弦相似度）
        """
        batch_size, num_views, feat_dim = view_feats.shape
        # 处理边界情况
        if batch_size < 1 or num_views < 2:
            zero_loss = torch.tensor(0.0, device=view_feats.device, dtype=view_feats.dtype)
            zero_loss.requires_grad_(True)
            return zero_loss
        
        # 特征归一化
        view_feats_norm = F.normalize(view_feats, p=2, dim=-1)
        all_similarities = []
        
        # 计算同一对象不同视图之间的余弦相似度
        for obj_idx in range(batch_size):
            obj_view_feats = view_feats_norm[obj_idx]
            # 遍历所有视图对（i,j）且i<j，避免重复计算
            for i in range(num_views):
                for j in range(i + 1, num_views):
                    # 计算视图i和视图j之间的余弦相似度
                    cos_sim = F.cosine_similarity(obj_view_feats[i:i + 1], obj_view_feats[j:j + 1], dim=-1)
                    all_similarities.append(cos_sim)
        
        # 返回1-平均相似度作为损失
        return (1 - torch.stack(all_similarities).mean()) if all_similarities else torch.tensor(0.0,
                                                                                               device=view_feats.device,
                                                                                               requires_grad=True)
    
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