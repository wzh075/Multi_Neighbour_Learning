#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MultiViewEncoder(nn.Module):
    """共享的视点特征提取器（优化特征分布）"""
    def __init__(self, feat_dim=256, feat_activation_scaling=1.0):
        super().__init__()
        self.feat_activation_scaling = feat_activation_scaling
        # 使用weights参数替代deprecated的pretrained参数
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 创建特征提取器，只包含需要的层
        # 保留卷积层和池化层，但不包含最后的全连接层
        self.feature_extractor = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 特征投影层 - 优化特征分布
        self.projection = nn.Sequential(
            nn.Linear(512, feat_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(feat_dim),  # 添加LayerNorm改善特征分布
            nn.Linear(feat_dim, feat_dim)
        )
        # 额外的激活缩放层
        self.activation_scaling = nn.Parameter(torch.tensor(feat_activation_scaling))
        # 添加特征缩放和偏置参数以优化特征分布
        # 增加特征缩放系数并设置非零偏置，避免特征接近全0
        self.feature_scale = nn.Parameter(torch.ones(feat_dim) * 4.0 * feat_activation_scaling)  # 增加初始放大倍数到4倍，并应用特征激活缩放
        self.feature_bias = nn.Parameter(torch.ones(feat_dim) * 0.5)  # 设置非零偏置，提高特征均值
    
    def forward(self, x):
        """输入: (batch, num_images, C, H, W)"""
        batch_size, num_images = x.shape[:2]
        # 获取剩余维度并构建reshape参数
        remaining_dims = list(x.shape[2:])
        # 手动构建reshape参数列表
        reshape_args = [-1] + remaining_dims
        x = x.reshape(*reshape_args)  # (batch*num_images, C, H, W)
        features = self.feature_extractor(x)  # (batch*num_images, 512)
        features = self.projection(features)  # (batch*num_images, feat_dim)
        # 应用激活缩放
        features = features * self.activation_scaling
        # 应用特征缩放和偏置优化分布
        features = features * self.feature_scale + self.feature_bias
        return features.reshape(batch_size, num_images, -1)  # (batch, num_images, feat_dim)


class CrossAttentionFusion(nn.Module):
    """交叉注意力模块：融合3个视点特征生成中间特征"""
    def __init__(self, feat_dim, num_heads=None):
        super().__init__()
        
        # 自动计算合适的num_heads，确保能被feat_dim整除
        if num_heads is None:
            # 寻找最大的能整除feat_dim的头数，不超过8
            for h in [8, 4, 2, 1]:
                if feat_dim % h == 0:
                    num_heads = h
                    break
        else:
            # 检查用户提供的num_heads是否合法
            if feat_dim % num_heads != 0:
                raise ValueError(f"feat_dim={feat_dim}必须能被num_heads={num_heads}整除")
        
        # 多头注意力层（查询、键、值维度均为feat_dim）
        self.attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True  # 启用batch_first模式，输入形状为(batch, seq_len, dim)
        )
        # 记录实际使用的头数以便调试
        self.num_heads = num_heads
        
    def forward(self, view_feats):
        """
        输入: view_feats (batch, 3, feat_dim) 包含3个视点的特征[v1, v2, v3]
        输出: mid_feat (batch, feat_dim) 融合后的中间特征
        """
        batch_size = view_feats.shape[0]
        cross_attn_outputs = []
        
        # 对每个视点特征，用另外两个视点作为键和值计算交叉注意力
        for i in range(3):  # i=0,1,2 分别对应3个视点
            # 当前视点作为查询 (batch, 1, feat_dim)
            query = view_feats[:, i:i+1, :]  # 取第i个视点，保持seq_len=1
            
            # 另外两个视点作为键和值 (batch, 2, feat_dim)
            others = [view_feats[:, j:j+1, :] for j in range(3) if j != i]
            key = torch.cat(others, dim=1)  # 拼接另外两个视点
            value = key  # 键值相同
            
            # 计算注意力：(batch, 1, feat_dim)
            attn_output, _ = self.attention(query, key, value)
            cross_attn_outputs.append(attn_output.squeeze(1))  # 移除seq_len维度
        
        # 3个交叉注意力输出取平均作为中间特征
        mid_feat = torch.stack(cross_attn_outputs, dim=1).mean(dim=1)  # (batch, feat_dim)
        return mid_feat


class GlobalMappingNetwork(nn.Module):
    """全局尺度映射网络：从中间特征生成全局特征"""
    def __init__(self, feat_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
    
    def forward(self, mid_feat):
        """
        输入: mid_feat (batch, feat_dim) 交叉注意力输出的中间特征
        输出: global_feat (batch, feat_dim) 全局尺度特征
        """
        return self.mapping(mid_feat)


class ViewGlobalFusion(nn.Module):
    """视点-全局融合模块：结合全局特征和视点特征生成对象级特征（优化特征分布）"""
    def __init__(self, feat_dim, num_heads=None, feat_activation_scaling=1.0):
        super().__init__()
        self.feat_activation_scaling = feat_activation_scaling
        
        # 自动计算合适的num_heads，确保能被feat_dim整除
        if num_heads is None:
            # 寻找最大的能整除feat_dim的头数，不超过8
            for h in [8, 4, 2, 1]:
                if feat_dim % h == 0:
                    num_heads = h
                    break
        else:
            # 检查用户提供的num_heads是否合法
            if feat_dim % num_heads != 0:
                raise ValueError(f"feat_dim={feat_dim}必须能被num_heads={num_heads}整除")
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # 记录实际使用的头数以便调试
        self.num_heads = num_heads
        # 融合后维度压缩层，增加LayerNorm优化特征分布
        self.fusion_proj = nn.Sequential(
            nn.Linear(2 * feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),  # 添加LayerNorm
            nn.LeakyReLU(negative_slope=0.1)  # 使用LeakyReLU
        )
        # 添加调试标志
        self.debug = False
        # 优化特征分布的参数
        self.feature_scale = nn.Parameter(torch.ones(feat_dim) * feat_activation_scaling)
        self.feature_bias = nn.Parameter(torch.zeros(feat_dim))
        # 额外的激活缩放层
        self.activation_scaling = nn.Parameter(torch.tensor(feat_activation_scaling))
        # 新增全局特征和视点特征的预处理层
        self.global_preprocess = nn.Linear(feat_dim, feat_dim)
        self.view_preprocess = nn.Linear(feat_dim, feat_dim)
    
    def forward(self, global_feat, view_feats):
        """
        融合全局特征和视点特征
        
        参数:
        - global_feat: 全局特征 [batch, feat_dim]
        - view_feats: 视点特征 [batch, num_views, feat_dim]
        
        返回:
        - obj_feat: 对象级特征 [batch, feat_dim]
        """
        batch_size, num_views, _ = view_feats.shape
        
        # 预处理全局特征和视点特征以改善分布
        global_feat = self.global_preprocess(global_feat)  # 预处理全局特征
        
        # 应用特征缩放和偏置优化分布
        global_feat = global_feat * self.feature_scale + self.feature_bias
        
        # 对每个视点特征进行预处理和缩放
        view_feats_reshaped = view_feats.reshape(-1, view_feats.size(-1))  # [batch*num_views, feat_dim]
        view_feats_reshaped = self.view_preprocess(view_feats_reshaped)
        view_feats_reshaped = view_feats_reshaped * self.feature_scale + self.feature_bias
        view_feats = view_feats_reshaped.view(batch_size, num_views, -1)  # [batch, num_views, feat_dim]
        
        # 全局特征作为查询（扩展seq_len维度）
        query = global_feat.unsqueeze(1)  # (batch, 1, feat_dim)
        
        # 视点特征作为键和值
        key = view_feats  # (batch, num_views, feat_dim)
        value = view_feats
        
        # 计算注意力：全局特征关注重要的视点特征
        attn_output, attn_weights = self.attention(query, key, value)  # (batch, 1, feat_dim)
        attended_view_feat = attn_output.squeeze(1)  # (batch, feat_dim)
        
        # 拼接全局特征和注意力加权后的视点特征，压缩维度
        fused = torch.cat([global_feat, attended_view_feat], dim=1)  # (batch, 2*feat_dim)
        
        # 应用融合投影
        obj_feat = self.fusion_proj(fused)  # (batch, feat_dim)
        
        # 再次应用特征缩放以确保特征分布合理
        obj_feat = obj_feat * self.feature_scale * 0.5 + self.feature_bias * 0.5
        
        # 移除调试打印以避免内存泄漏
        
        return obj_feat


class MultiViewRetrievalModel(nn.Module):
    def __init__(self, num_classes, feat_dim=256, feat_activation_scaling=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.feat_activation_scaling = feat_activation_scaling
        
        # 共享的视点编码器（保持不变）
        self.view_encoder = MultiViewEncoder(feat_dim, feat_activation_scaling)
        
        # 新增模块：交叉注意力融合、全局映射、视点-全局融合
        self.cross_attn_fusion = CrossAttentionFusion(feat_dim)
        self.global_mapping = GlobalMappingNetwork(feat_dim)
        self.view_global_fusion = ViewGlobalFusion(feat_dim, feat_activation_scaling=feat_activation_scaling)
        
        # 禁用调试模式以减少输出
        self.view_global_fusion.debug = False
        
        # 初始化模型权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 确保权重不会全为零，使用kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                # 手动初始化MultiheadAttention的权重
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
        
        # 特别检查并初始化ViewGlobalFusion的fusion_proj层
    
    def forward(self, view1, view2, view3):
        # 1. 提取每个视点组的特征
        # view1, view2, view3的形状是(batch_size, num_images_per_view, C, H, W)
        # self.view_encoder返回的形状是(batch_size, num_images_per_view, feat_dim)
        feat1 = self.view_encoder(view1)  # (batch, num_images_per_view, feat_dim)
        feat2 = self.view_encoder(view2)
        feat3 = self.view_encoder(view3)
        
        # 2. 为InfoNCE损失准备正确的view_feats结构
        # 我们需要将每个对象的三个视点分别作为独立样本处理
        # 这里我们将三个视点的特征直接堆叠，保持batch维度不变
        # 最终view_feats形状为(batch, 3, feat_dim)，每个batch元素包含一个对象的三个视点特征
        # 但我们需要确保InfoNCE损失能够识别同一对象的不同视点作为正样本
        view_feats = torch.stack([
            feat1.mean(1),  # 对每个视点组的多个图像取平均
            feat2.mean(1),
            feat3.mean(1)
        ], dim=1)  # (batch, 3, feat_dim)
        
        # 特征增强：在InfoNCE计算前进行特征偏移，提高均值
        view_feats = view_feats * 1.5 * self.feat_activation_scaling + 0.3  # 增加额外的缩放和偏移，应用特征激活缩放
        
        # 2. 交叉注意力融合生成中间特征
        mid_feat = self.cross_attn_fusion(view_feats)  # (batch, feat_dim)
        
        # 3. 全局映射网络生成全局特征
        global_feat = self.global_mapping(mid_feat)  # (batch, feat_dim)
        
        # 4. 视点-全局融合生成对象级特征
        obj_feat = self.view_global_fusion(global_feat, view_feats)  # (batch, feat_dim)
        
        # 调整归一化策略：只对需要用于InfoNCE损失的特征进行归一化
        # 保持对象特征为原始分布，在损失函数中根据需要进行归一化
        normalized_obj_feat = F.normalize(obj_feat, p=2, dim=-1)
        normalized_mid_feat = F.normalize(mid_feat, p=2, dim=-1)
        normalized_global_feat = F.normalize(global_feat, p=2, dim=-1)

        # 移除调试打印以避免内存泄漏

        return {
            'view_feats': view_feats,                  # (batch, 3, feat_dim) 原始视图特征
            'mid_feat': normalized_mid_feat,           # (batch, feat_dim) 归一化的中间特征
            'global_feat': normalized_global_feat,     # (batch, feat_dim) 归一化的全局特征
            'obj_feat': obj_feat,                      # (batch, feat_dim) 原始对象特征（不再归一化）
            'raw_mid_feat': mid_feat,                  # (batch, feat_dim) 原始中间特征
            'raw_global_feat': global_feat,            # (batch, feat_dim) 原始全局特征
            'raw_obj_feat': obj_feat                   # (batch, feat_dim) 原始对象特征
        }
    
    def encode_single_view(self, view):
        """编码单个视点组（保持不变）"""
        feat = self.view_encoder(view)  # (batch, 5, feat_dim)
        return feat.mean(1)  # (batch, feat_dim)