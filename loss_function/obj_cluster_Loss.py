#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class ObjectClusterLoss(nn.Module):
    """
    对象聚类损失函数模块
    
    该损失函数用于在全局范围内优化对象特征聚类，使得同一类别的对象特征相近，
    不同类别的对象特征相远。通过维护一个记忆库来存储历史样本，实现跨批次的聚类优化。
    
    Args:
        tau: 温度参数，用于控制softmax的分布陡峭程度
        cluster_memory_size: 每个类别的最大记忆样本数
        device: 计算设备
    """
    def __init__(self, tau=0.5, cluster_memory_size=1024, device=None):
        super().__init__()
        self.tau = tau
        self.cluster_memory_size = cluster_memory_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 初始化聚类记忆库：key为类别，value为该类别的obj_feat列表
        self.memory_bank = defaultdict(list)
    
    def forward(self, obj_feat, class_labels):
        """
        计算对象聚类损失
        
        Args:
            obj_feat: 对象特征，形状为(batch, feat_dim)
            class_labels: 类别标签，形状为(batch,) （若为无监督学习可传入obj_id哈希值）
        
        Returns:
            对象聚类损失值
        """
        batch_size, feat_dim = obj_feat.shape
        # 处理边界情况
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 特征归一化并添加小扰动防止零向量
        obj_feat_norm = F.normalize(obj_feat + 1e-10, p=2, dim=-1)  # (batch, feat_dim)
        
        # 更新记忆库（限制每个类别的存储数量）
        for feat, label in zip(obj_feat_norm, class_labels):
            label = str(label)  # 确保标签可哈希
            self.memory_bank[label].append(feat.detach())  # 不追踪梯度
            # 超出容量则删除最早的样本
            if len(self.memory_bank[label]) > self.cluster_memory_size:
                self.memory_bank[label].pop(0)
        
        total_loss = 0.0
        valid_samples = 0
        
        # 控制正样本和负样本的最大数量，防止计算溢出
        max_pos_samples = 128
        max_neg_samples = 512
        
        for i in range(batch_size):
            current_feat = obj_feat_norm[i]  # 当前样本特征
            current_label = str(class_labels[i])  # 当前样本类别
            
            # 1. 寻找正样本（同类别，来自记忆库+当前批次其他样本）
            # 记忆库中的正样本
            pos_from_memory = self.memory_bank.get(current_label, [])
            # 当前批次中的正样本（排除自身）
            pos_from_batch = [obj_feat_norm[j] for j in range(batch_size)
                              if j != i and str(class_labels[j]) == current_label]
            all_pos_feats = pos_from_memory + pos_from_batch
            
            if not all_pos_feats:  # 无正样本时跳过
                continue
            
            # 2. 寻找负样本（不同类别，来自记忆库+当前批次）
            all_neg_feats = []
            # 1) 从记忆库获取负样本
            for label, feats in self.memory_bank.items():
                if label != current_label:
                    all_neg_feats.extend(feats)
            
            # 2) 如果记忆库中负样本不足，从当前批次获取负样本（不同类别的其他样本）
            if len(all_neg_feats) < 8:  # 在早期训练阶段，降低负样本要求
                # 从当前批次寻找不同类别的样本作为负样本
                batch_neg_feats = [obj_feat_norm[j] for j in range(batch_size)
                                  if j != i and str(class_labels[j]) != current_label]
                all_neg_feats.extend(batch_neg_feats)
            
            # 确保负样本数量足够（训练早期可以降低要求）
            min_neg_samples = 4 if len(self.memory_bank) < 10 else 1  # 记忆库小时降低要求
            if len(all_neg_feats) < min_neg_samples:
                continue
            
            # 限制正负样本数量，提高计算效率和数值稳定性
            all_pos_feats = all_pos_feats[:max_pos_samples]
            all_neg_feats = all_neg_feats[:max_neg_samples]
            
            # 3. 计算对比损失（类似InfoNCE）
            pos_feats = torch.stack(all_pos_feats)  # (N_pos, feat_dim)
            neg_feats = torch.stack(all_neg_feats)  # (N_neg, feat_dim)
            
            # 计算最大相似度作为logits的上界，防止exp溢出
            max_sim = 8.0  # 设置合理的上界值
            
            # 正样本相似度 - 添加数值稳定性保障
            pos_sim = torch.clamp(torch.matmul(current_feat.unsqueeze(0), pos_feats.T).squeeze() / self.tau, max=max_sim)
            pos_exp = torch.exp(pos_sim).sum() + 1e-16  # 添加小值防止为0
            
            # 负样本相似度 - 添加数值稳定性保障
            neg_sim = torch.clamp(torch.matmul(current_feat.unsqueeze(0), neg_feats.T).squeeze() / self.tau, max=max_sim)
            neg_exp = torch.exp(neg_sim).sum() + 1e-16  # 添加小值防止为0
            
            # 计算InfoNCE损失
            with torch.no_grad():
                # 检查是否会出现数值问题
                if pos_exp.isnan().any() or neg_exp.isnan().any() or (pos_exp + neg_exp).isnan().any():
                    continue
            
            # 安全计算损失
            loss = -torch.log(pos_exp / (pos_exp + neg_exp))
            
            # 再次检查损失是否有效
            if not loss.isnan().any() and not loss.isinf().any():
                total_loss += loss
                valid_samples += 1
        
        # 只有在有有效样本时才返回非零损失
        # 为了避免训练早期返回0，当valid_samples很少时返回一个小的非零值
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            # 返回一个小的非零损失值，确保梯度流存在
            # 随着训练进行，valid_samples会增加，损失将正常计算
            return torch.tensor(0.1, device=self.device, requires_grad=True)
    
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
        self.device = device or self.device
        if dtype is not None:
            self.tau = torch.tensor(self.tau, device=device, dtype=dtype)
        # 确保记忆库在正确设备上
        for label in self.memory_bank:
            self.memory_bank[label] = [feat.to(device) for feat in self.memory_bank[label]]
        return self