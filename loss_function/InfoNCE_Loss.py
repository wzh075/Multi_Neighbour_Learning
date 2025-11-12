#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比损失函数模块
    
    用于多视图学习中的对比学习，将同一对象的不同视图视为正样本对，
    不同对象的视图视为负样本对，优化特征分布
    
    Args:
        tau: 温度参数，用于控制softmax的分布陡峭程度
    """
    def __init__(self, tau=0.5):  # 增大温度参数，使分布更平滑，避免损失过早趋近于0
        super().__init__()
        self.tau = tau
        # 可学习的温度参数，便于动态调整
        self.learnable_tau = nn.Parameter(torch.tensor(self.tau))
        # 批次内差异增强参数，降低初始权重
        self.batch_diversity_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, view_feats, obj_ids):
        """
        计算InfoNCE损失
        
        Args:
            view_feats: 视点特征，形状为(batch, num_views, feat_dim)
            obj_ids: 对象ID，用于标识正样本对
        
        Returns:
            平均InfoNCE损失值
        """
        batch_size, num_views, feat_dim = view_feats.shape
        # 处理边界情况
        if batch_size < 1 or num_views < 2:
            zero_loss = torch.tensor(0.0, device=view_feats.device, dtype=view_feats.dtype)
            zero_loss.requires_grad_(True)
            return zero_loss
            
        # 特征归一化，确保特征在单位球面上
        view_feats_norm = F.normalize(view_feats, p=2, dim=-1)
        all_losses = []
        
        # 使用可学习的温度参数，确保温度为正数，扩大范围以允许更平滑的分布
        effective_tau = torch.clamp(self.learnable_tau, min=0.1, max=2.0)
        
        # 计算批次内所有视图的特征
        all_views_flat = view_feats_norm.reshape(-1, feat_dim)
        
        # 对每个对象计算损失
        for obj_idx in range(batch_size):
            obj_view_feats = view_feats_norm[obj_idx]
            # 计算同一对象不同视图之间的相似度矩阵
            obj_sim_matrix = torch.matmul(obj_view_feats, obj_view_feats.T) / effective_tau
            
            # 对每个视图计算InfoNCE损失
            for view_idx in range(num_views):
                # 正样本是同一对象的其他视图
                pos_mask = torch.ones(num_views, dtype=torch.bool)
                pos_mask[view_idx] = False
                pos_exp_sum = torch.exp(obj_sim_matrix[view_idx, pos_mask]).sum()
                
                # 负样本是其他对象的所有视图
                other_obj_feats = view_feats_norm[torch.arange(batch_size) != obj_idx].reshape(-1, feat_dim)
                neg_exp_sum = torch.exp(torch.matmul(obj_view_feats[view_idx:view_idx + 1],
                                                     other_obj_feats.T).squeeze() / effective_tau).sum() if len(
                    other_obj_feats) > 0 else torch.tensor(1.0, device=view_feats.device)
                
                # 计算InfoNCE损失
                denominator = pos_exp_sum + neg_exp_sum + 1e-16
                view_loss = -torch.log(pos_exp_sum / denominator + 1e-16)
                all_losses.append(view_loss)
        
        # 计算平均损失
        avg_loss = torch.stack(all_losses).mean() if all_losses else torch.tensor(0.0, device=view_feats.device,
                                                                                 requires_grad=True)
        
        # 添加批次内对象间对比损失，增强融合特征的批次差异
        if batch_size > 1:
            # 计算每个对象的平均视图特征
            obj_mean_feats = view_feats_norm.mean(dim=1)  # (batch, feat_dim)
            
            # 计算批次内对象间的相似度矩阵
            obj_sim_matrix = torch.matmul(obj_mean_feats, obj_mean_feats.T) / effective_tau
            
            # 对角线为同一对象，设为0
            obj_sim_matrix.diagonal().zero_()
            
            # 计算批次内对象间的对比损失，鼓励不同对象之间的特征差异
            # 使用较小的温度参数增强区分度
            batch_diversity_loss = -torch.log(torch.exp(-obj_sim_matrix).mean() + 1e-16)
            
            # 将批次内差异损失添加到总损失中
            avg_loss += torch.clamp(self.batch_diversity_weight, min=0.0, max=1.0) * batch_diversity_loss
        
        # 增强的特征分布正则化，直接防止特征坍塌到零点
        # 1. 特征均值正则化：鼓励特征均值远离零
        target_mean = torch.ones(feat_dim, device=view_feats.device) * 0.1  # 目标均值设为0.1
        mean_reg = 0.2 * F.mse_loss(view_feats_norm.mean(dim=0).mean(dim=0), target_mean)
        
        # 2. 增强的特征方差正则化：确保特征具有足够的差异性
        # 计算每个特征维度的方差
        feat_var = view_feats_norm.var(dim=0).mean(dim=0)
        # 目标方差设为更合理的值，避免特征过度分散
        target_var = torch.ones(feat_dim, device=view_feats.device) * 0.05  # 降低目标方差
        # 使用Huber损失更鲁棒地优化方差，降低权重
        var_reg = 0.1 * F.huber_loss(feat_var, target_var, delta=0.1)
        
        # 3. 批次内特征离散度增强：确保同一批次内不同对象的特征有足够差异
        if batch_size > 1:
            # 计算批次内对象间特征的最大相似度
            obj_mean_feats = view_feats_norm.mean(dim=1)
            obj_sim_matrix = torch.matmul(obj_mean_feats, obj_mean_feats.T)
            # 移除对角线元素（自相似度）
            obj_sim_matrix.fill_diagonal_(0)
            # 计算最大相似度
            max_cross_sim = obj_sim_matrix.max()
            # 降低相似度正则化权重，避免过度惩罚相似特征
            similarity_reg = 0.2 * F.relu(max_cross_sim - 0.7)  # 提高阈值至0.7，允许更多相似性
        else:
            similarity_reg = torch.tensor(0.0, device=view_feats.device, requires_grad=True)
        
        # 4. 特征稀疏性正则化：防止特征过度集中，降低权重
        sparsity_reg = 0.01 * torch.mean(torch.abs(view_feats_norm))
        
        # 将正则化项添加到总损失中，按顺序添加并确保总损失不会过小
        avg_loss = avg_loss + mean_reg + var_reg + similarity_reg + sparsity_reg
        
        # 确保损失有一个最小下限，防止损失为0
        min_loss = torch.tensor(0.01, device=view_feats.device, dtype=avg_loss.dtype)
        avg_loss = torch.max(avg_loss, min_loss)
        
        # 移除调试打印以避免内存泄漏
        
        return avg_loss
    
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
        if dtype is not None:
            self.tau = torch.tensor(self.tau, device=device, dtype=dtype)
            # 确保可学习的温度参数也更新数据类型
            self.learnable_tau.data = self.learnable_tau.data.to(device=device, dtype=dtype)
        return self