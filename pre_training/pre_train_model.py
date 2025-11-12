#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models

class PreMVCNN(nn.Module):
    """
    预训练MVCNN模型，用于class级别的分类任务
    基于MultiViewEncoder结构，但添加了分类头
    """
    def __init__(self, num_classes, feat_dim=256):
        super().__init__()
        # 基础模型部分与MultiViewEncoder保持一致
        base_model = models.resnet18(pretrained=True)
        base_layers = list(base_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*base_layers)
        self.feature_extractor.add_module('adaptive_avg_pool', nn.AdaptiveAvgPool2d(1))
        self.feature_extractor.add_module('flatten', nn.Flatten())
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.Linear(512, feat_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim)
        )
        
        # 优化特征分布的参数
        self.feature_scale = nn.Parameter(torch.ones(feat_dim) * 4.0)
        self.feature_bias = nn.Parameter(torch.ones(feat_dim) * 0.5)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        x: (batch_size, num_views, C, H, W) - 多视图输入
        """
        batch_size, num_views, C, H, W = x.shape
        
        # 将多个视图合并到batch维度
        x = x.reshape(-1, C, H, W)  # (batch*num_views, C, H, W)
        
        # 提取特征
        features = self.feature_extractor(x)  # (batch*num_views, 512)
        features = self.projection(features)  # (batch*num_views, feat_dim)
        
        # 应用特征缩放和偏置
        features = features * self.feature_scale + self.feature_bias
        
        # 重塑回(batch, num_views, feat_dim)
        features = features.reshape(batch_size, num_views, -1)
        
        # 对多个视图的特征取平均，获得对象级特征
        obj_features = features.mean(dim=1)  # (batch, feat_dim)
        
        # 分类预测
        logits = self.classifier(obj_features)  # (batch, num_classes)
        
        return {
            'logits': logits,
            'features': obj_features,
            'view_features': features
        }
    
    def get_feature_extractor(self):
        """获取特征提取器部分，用于迁移学习"""
        feature_extractor = nn.Sequential(
            self.feature_extractor,
            self.projection
        )
        # 将特征缩放和偏置参数也包含进去
        feature_extractor.feature_scale = self.feature_scale
        feature_extractor.feature_bias = self.feature_bias
        return feature_extractor
    
    def load_into_multiview_encoder(self, multiview_encoder):
        """将预训练权重加载到MultiViewEncoder中"""
        # 加载特征提取器权重
        for m_pretrained, m_target in zip(
            self.feature_extractor.modules(), 
            multiview_encoder.feature_extractor.modules()
        ):
            if isinstance(m_pretrained, nn.Conv2d) or isinstance(m_pretrained, nn.Linear):
                if hasattr(m_target, 'weight'):
                    m_target.weight.data.copy_(m_pretrained.weight.data)
                if hasattr(m_target, 'bias') and m_target.bias is not None:
                    m_target.bias.data.copy_(m_pretrained.bias.data)
        
        # 加载投影层权重
        for m_pretrained, m_target in zip(
            self.projection.modules(), 
            multiview_encoder.projection.modules()
        ):
            if isinstance(m_pretrained, nn.Conv2d) or isinstance(m_pretrained, nn.Linear):
                if hasattr(m_target, 'weight'):
                    m_target.weight.data.copy_(m_pretrained.weight.data)
                if hasattr(m_target, 'bias') and m_target.bias is not None:
                    m_target.bias.data.copy_(m_pretrained.bias.data)
        
        # 加载特征缩放和偏置参数
        multiview_encoder.feature_scale.data.copy_(self.feature_scale.data)
        multiview_encoder.feature_bias.data.copy_(self.feature_bias.data)
        
        print("预训练权重已成功加载到MultiViewEncoder中")
        return multiview_encoder