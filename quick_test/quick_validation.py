#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入必要的模块
from training.train import train_model  # 修复导入，使用正确的函数名
from models.MultiView_Retrieval_Model import MultiViewRetrievalModel
from loss_function.InfoNCE_Loss import InfoNCELoss
from loss_function.view_similarity_Loss import ViewSimilarityLoss
from loss_function.global_consistency_Loss import GlobalConsistencyLoss
from loss_function.obj_cluster_Loss import ObjectClusterLoss
from tools.Multi_Neighbour_View_Dataloader import MultiViewDataset

# 设置日志
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'quick_validation_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('quick_validation')

# 加载配置
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 初始化模型
def init_model():
    # 从配置中获取类别数量，这里先使用一个默认值100
    # 在实际运行中可能需要从数据集中动态获取
    num_classes = 100
    
    model = MultiViewRetrievalModel(
        num_classes=num_classes,
        feat_dim=128
    )
    return model

# 自定义损失函数包装器，处理train.py中传递的所有参数
class CustomLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.infonce_loss = InfoNCELoss(tau=0.07)
        self.view_similarity_loss = ViewSimilarityLoss()
        self.global_consistency_loss = GlobalConsistencyLoss()
        self.obj_cluster_loss = ObjectClusterLoss(tau=0.07)
    
    def forward(self, view_feats, mid_feat, global_feat, obj_ids, obj_feat=None, class_labels=None, **kwargs):
        # 计算所有损失
        infonce_loss_val = self.infonce_loss(view_feats, obj_ids)
        view_sim_loss_val = self.view_similarity_loss(view_feats)
        
        # 计算全局一致性损失
        global_cons_loss_val = self.global_consistency_loss(view_feats, global_feat)
        
        # 计算对象聚类损失，如果提供了必要的参数
        if obj_feat is not None and class_labels is not None:
            obj_clust_loss_val = self.obj_cluster_loss(obj_feat, class_labels)
        else:
            obj_clust_loss_val = torch.tensor(0.0, device=infonce_loss_val.device)
        
        # 返回所有损失项的字典
        return {
            'infonce_loss': infonce_loss_val,
            'view_similarity_loss': view_sim_loss_val,
            'global_consistency_loss': global_cons_loss_val,
            'obj_cluster_loss': obj_clust_loss_val
        }

# 初始化损失函数
def init_loss():
    criterion = CustomLossWrapper()
    return criterion

# 数据加载
def load_dataset(dataset_path, split='train'):
    dataset = MultiViewDataset(
        root_dir=dataset_path,
        split=split,
        num_views=3
    )
    return dataset

# 特征提取
def extract_features(model, dataset, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    features = []
    labels = []
    obj_ids = []
    
    # 创建简单的数据加载器
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            view1, view2, view3 = images[:, 0], images[:, 1], images[:, 2]
            
            # 提取特征
            outputs = model(view1, view2, view3)
            
            # 保存特征和标签
            features.append(outputs['global_feat'].cpu().numpy())
            labels.extend(batch['label'].tolist())
            obj_ids.extend(batch['obj_id'])
    
    # 合并特征
    features = np.concatenate(features, axis=0)
    
    return features, labels, obj_ids

# 简单的检索评估
def evaluate_retrieval(features, labels, topk=10):
    # 添加调试信息
    print(f"特征形状: {features.shape}")
    print(f"标签数量: {len(labels)}")
    print(f"标签样本: {labels[:5]}")
    
    # 检查类别分布
    unique_labels = np.unique(labels)
    print(f"唯一类别数量: {len(unique_labels)}")
    print(f"类别分布: {dict(zip(unique_labels, np.bincount(labels)))}")
    
    # 特征归一化（这很重要，确保相似度计算正确）
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-16)
    
    # 计算相似度矩阵
    similarity_matrix = features_norm @ features_norm.T
    
    # 添加相似度矩阵统计信息
    print(f"相似度矩阵形状: {similarity_matrix.shape}")
    print(f"相似度矩阵样本值（前5x5）:")
    print(similarity_matrix[:5, :5])
    print(f"相似度范围: {similarity_matrix.min():.4f} - {similarity_matrix.max():.4f}")
    
    # 对角线设为0（避免检索到自身）
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    # 计算召回率
    recalls = {k: 0 for k in [1, 5, 10]}
    
    for i in range(len(labels)):
        query_label = labels[i]
        sims = similarity_matrix[i]
        
        # 按相似度排序
        top_indices = np.argsort(-sims)
        
        # 调试：打印前5个最相似的索引和对应的标签
        if i < 3:  # 只打印前3个查询的信息
            print(f"查询 {i} (标签: {query_label}) 的top5相似结果:")
            for j in range(5):
                if j < len(top_indices):
                    print(f"  索引: {top_indices[j]}, 标签: {labels[top_indices[j]]}, 相似度: {sims[top_indices[j]]:.4f}")
        
        # 计算各k值的召回率
        for k in recalls.keys():
            top_k_indices = top_indices[:k]
            top_k_labels = [labels[j] for j in top_k_indices]
            if query_label in top_k_labels:
                recalls[k] += 1
    
    # 计算平均召回率
    for k in recalls:
        recalls[k] = recalls[k] / len(labels)
    
    return recalls

# 主函数
def main(dataset_path, output_dir):
    # 设置日志
    log_dir = os.path.join(output_dir, 'logs')
    logger = setup_logger(log_dir)
    
    # 加载配置
    config = load_config()
    
    logger.info("开始快速验证流程")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"输出目录: {output_dir}")
    
    try:
        # 1. 加载数据集
        logger.info("加载数据集...")
        train_dataset = load_dataset(dataset_path, 'train')
        logger.info(f"训练数据集加载完成，包含 {len(train_dataset)} 个样本")
        
        # 2. 初始化模型
        logger.info("初始化模型...")
        model = init_model()
        logger.info("模型初始化完成")
        
        # 3. 初始化损失函数
        logger.info("初始化损失函数...")
        criterion = init_loss()
        logger.info("损失函数初始化完成")
        
        # 4. 训练模型
        logger.info("开始训练模型...")
        model_save_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 使用配置中的训练参数
        epochs = config['training'].get('epochs', 5)
        batch_size = config['training'].get('batch_size', 4)
        learning_rate = config['training'].get('learning_rate', 0.001)
        
        trained_model = train_model(
            model=model,
            dataset=train_dataset,
            criterion=criterion,
            epochs=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            save_dir=model_save_dir
        )
        logger.info("模型训练完成")
        
        # 5. 提取特征
        logger.info("提取特征...")
        features, labels, obj_ids = extract_features(
            trained_model, 
            train_dataset, 
            batch_size=config['feature_extraction'].get('batch_size', 4)
        )
        logger.info(f"特征提取完成，共提取 {len(features)} 个样本的特征")
        
        # 保存特征
        features_dir = os.path.join(output_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)
        features_path = os.path.join(features_dir, 'extracted_features.json')
        
        features_data = {
            'features': features.tolist(),
            'labels': labels,
            'obj_ids': obj_ids
        }
        
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features_data, f)
        logger.info(f"特征已保存至: {features_path}")
        
        # 6. 评估检索性能
        logger.info("评估检索性能...")
        recalls = evaluate_retrieval(features, labels)
        logger.info("检索评估完成")
        
        # 打印检索结果
        logger.info("检索性能结果:")
        for k, recall in recalls.items():
            logger.info(f"Recall@{k}: {recall:.4f}")
        
        logger.info("快速验证流程成功完成!")
        return True
        
    except Exception as e:
        logger.error(f"验证过程中发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多视图检索系统快速验证脚本')
    parser.add_argument('--dataset-path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output-dir', type=str, required=True, help='输出目录')
    args = parser.parse_args()
    
    success = main(args.dataset_path, args.output_dir)
    sys.exit(0 if success else 1)