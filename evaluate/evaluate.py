#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import numpy as np
import h5py
from torchvision import transforms
from tqdm import tqdm
from torch import nn

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模型和数据加载器
from models.MultiView_Retrieval_Model import MultiViewRetrievalModel
from tools.Multi_Neighbour_View_Dataloader import MultiViewDataset

class FeatureDatabase:
    """
    特征数据库，用于存储和检索特征
    """
    def __init__(self, db_path):
        """
        :param db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.db = None
        self.object_features = {}
        self.view_features = {}
        self.class_map = {}
        
    def open(self):
        """打开数据库并加载特征"""
        if self.db is None:
            self.db = h5py.File(self.db_path, 'r')
        
        # 加载所有对象特征和视图特征
        for obj_id in tqdm(self.db.keys(), desc="加载特征数据库"):
            # 提取类别信息（假设obj_id格式为"class_obj"）
            class_name = obj_id.split('_')[0]
            self.class_map[obj_id] = class_name
            
            # 加载对象特征
            if 'obj_feat' in self.db[obj_id]:
                self.object_features[obj_id] = torch.from_numpy(self.db[obj_id]['obj_feat'][()])
            
            # 加载视图特征
            if 'view_features' in self.db[obj_id]:
                view_feats = torch.from_numpy(self.db[obj_id]['view_features'][()])
                for i, view_feat in enumerate(view_feats):
                    view_key = f"{obj_id}_view_{i}"
                    self.view_features[view_key] = view_feat
                    self.class_map[view_key] = class_name
        
        print(f"特征数据库加载完成: {len(self.object_features)}个对象特征, {len(self.view_features)}个视图特征")
    
    def close(self):
        """关闭数据库"""
        if self.db is not None:
            self.db.close()
            self.db = None
    
    def get_object_features(self):
        """获取所有对象特征"""
        return self.object_features
    
    def get_view_features(self):
        """获取所有视图特征"""
        return self.view_features
    
    def get_class_map(self):
        """获取对象/视图到类别的映射"""
        return self.class_map

def load_model(args, device):
    """
    加载用于评估的模型
    
    Args:
        args: 命令行参数
        device: 计算设备
    
    Returns:
        model: 加载好的模型
    """
    print("加载评估模型...")

    # 初始化模型
    model = MultiViewRetrievalModel(
        num_classes=args.num_classes,
        feat_dim=args.feat_dim
    )

    # 加载预训练权重
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # 检查是否是DataParallel保存的权重
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 严格加载权重，不允许参数不匹配
        model.load_state_dict(state_dict, strict=True)
        print(f"已成功加载模型权重: {args.model_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 将模型移动到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_images(images, transform, device):
    """
    预处理图像
    
    Args:
        images: 原始图像（可能已经是torch.Tensor类型）
        transform: 图像变换
        device: 计算设备
    
    Returns:
        processed_images: 预处理后的图像
    """
    # 检查images是否已经是tensor
    if isinstance(images, torch.Tensor):
        # 如果已经是tensor，直接返回并移动到设备
        return images.to(device)
    else:
        # 否则应用变换并移动到设备
        processed_images = []
        for img in images:
            processed_images.append(transform(img))
        
        return torch.stack(processed_images).to(device)

def single_viewgroup_retrieval(model, viewgroup_images, num_views, device, transform):
    """
    单视点组输入检索object
    
    Args:
        model: 评估模型
        viewgroup_images: 单视点组图像
        num_views: 训练时设置的视点组输入数量
        device: 计算设备
        transform: 图像变换
    
    Returns:
        object_feat: 对象特征
    """
    # 将单视点组通过复制扩展为训练设置的视点组输入数量
    expanded_viewgroups = [viewgroup_images] * num_views
    
    # 预处理所有视点组
    processed_viewgroups = []
    for viewgroup in expanded_viewgroups:
        processed_images = preprocess_images(viewgroup, transform, device)
        # 添加batch维度
        processed_images = processed_images.unsqueeze(0)  # [1, num_images_per_view, C, H, W]
        processed_viewgroups.append(processed_images)
    
    # 执行前向传播
    with torch.no_grad():
        # 根据训练时的输入格式，需要将每个视点组作为单独的参数传入
        outputs = model(*processed_viewgroups)
    
    return outputs['obj_feat']

def multi_viewgroup_retrieval(model, viewgroup_images_list, device, transform):
    """
    多视点组输入检索object
    
    Args:
        model: 评估模型
        viewgroup_images_list: 多视点组图像列表
        device: 计算设备
        transform: 图像变换
    
    Returns:
        object_feat: 对象特征
    """
    # 预处理所有视点组
    processed_viewgroups = []
    for viewgroup in viewgroup_images_list:
        processed_images = preprocess_images(viewgroup, transform, device)
        # 添加batch维度
        processed_images = processed_images.unsqueeze(0)  # [1, num_images_per_view, C, H, W]
        processed_viewgroups.append(processed_images)
    
    # 执行前向传播
    with torch.no_grad():
        outputs = model(*processed_viewgroups)
    
    return outputs['obj_feat']

def viewgroup_cross_retrieval(model, viewgroup_images, device, transform):
    """
    视点组之间互检索
    
    Args:
        model: 评估模型
        viewgroup_images: 视点组图像
        device: 计算设备
        transform: 图像变换
    
    Returns:
        view_feat: 视图特征
    """
    # 预处理图像
    processed_images = preprocess_images(viewgroup_images, transform, device)
    # 添加batch维度
    processed_images = processed_images.unsqueeze(0)  # [1, num_images_per_view, C, H, W]
    
    # 只使用view_encoder进行编码
    with torch.no_grad():
        view_feats = model.view_encoder(processed_images)
        
        # 平均池化得到该视点组的特征表示
        # 先在num_images_per_view维度上平均，然后在batch维度上平均
        view_feat = torch.mean(view_feats, dim=1)  # [1, feat_dim]
        view_feat = view_feat.squeeze(0)  # [feat_dim]
    
    return view_feat

def compute_recall(queries, database, class_map, top_k_list=[1, 5, 10]):
    """
    计算召回率
    
    Args:
        queries: 查询特征字典 {query_id: query_feat}
        database: 数据库特征字典 {db_id: db_feat}
        class_map: 类别映射字典 {id: class_name}
        top_k_list: 召回率的K值列表
    
    Returns:
        recall_results: 召回率结果字典
    """
    # 将数据库特征转换为张量
    db_ids = list(database.keys())
    db_feats = torch.stack([database[db_id] for db_id in db_ids])
    
    # 获取查询特征的设备（假设所有查询特征在同一设备上）
    if queries:
        device = next(iter(queries.values())).device
        db_feats = db_feats.to(device)
    
    # 计算所有查询的召回率
    recall_counts = {k: 0 for k in top_k_list}
    class_recall_counts = {k: 0 for k in top_k_list}
    total_queries = len(queries)
    
    for query_id, query_feat in tqdm(queries.items(), desc="计算召回率"):
        # 计算余弦相似度
        similarities = torch.nn.functional.cosine_similarity(
            query_feat.unsqueeze(0), db_feats, dim=1
        )
        
        # 获取相似度排序的索引
        _, indices = torch.sort(similarities, descending=True)
        # 去除batch维度，得到一维索引张量
        indices = indices.squeeze(0)
        
        # 获取查询的真实对象ID和类别
        query_obj_id = query_id.split('_view_')[0] if '_view_' in query_id else query_id
        query_class = class_map[query_id]
        
        # 检查每个top_k的召回情况
        for k in top_k_list:
            # 获取top-k的数据库ID
            top_k_indices = indices[:k]
            # 将索引张量转换为Python整数列表
            top_k_indices = top_k_indices.tolist()
            top_k_db_ids = [db_ids[idx] for idx in top_k_indices]
            
            # 检查是否召回了相同对象
            for db_id in top_k_db_ids:
                db_obj_id = db_id.split('_view_')[0] if '_view_' in db_id else db_id
                if db_obj_id == query_obj_id:
                    recall_counts[k] += 1
                    break
            
            # 检查是否召回了相同类别
            for db_id in top_k_db_ids:
                db_class = class_map[db_id]
                if db_class == query_class:
                    class_recall_counts[k] += 1
                    break
    
    # 计算召回率
    recall_results = {
        'object_recall': {k: recall_counts[k] / total_queries for k in top_k_list},
        'class_recall': {k: class_recall_counts[k] / total_queries for k in top_k_list}
    }
    
    return recall_results

def evaluate_single_viewgroup_retrieval(model, feature_db, dataset, num_views, device, transform):
    """
    评估单视点组输入检索object
    
    Args:
        model: 评估模型
        feature_db: 特征数据库
        dataset: 验证数据集
        num_views: 训练时设置的视点组数量
        device: 计算设备
        transform: 图像变换
    
    Returns:
        recall_results: 召回率结果
    """
    print("\n=== 单视点组输入检索object ===")
    
    # 构建查询集（只使用第一个视点组）
    queries = {}
    for i in tqdm(range(len(dataset)), desc="生成查询"):
        data = dataset[i]
        obj_id = data['obj_id']
        images = data['images']  # [num_views, num_images_per_view, C, H, W]
        
        # 只使用第一个视点组
        viewgroup_images = images[0]
        
        # 生成对象特征
        obj_feat = single_viewgroup_retrieval(model, viewgroup_images, num_views, device, transform)
        queries[obj_id] = obj_feat
    
    # 获取数据库
    database = feature_db.get_object_features()
    class_map = feature_db.get_class_map()
    
    # 计算召回率
    recall_results = compute_recall(queries, database, class_map)
    
    # 打印结果
    print("单视点组检索结果:")
    for k in sorted(recall_results['object_recall'].keys()):
        print(f"  Top-{k} Object Recall: {recall_results['object_recall'][k]:.4f}")
        print(f"  Top-{k} Class Recall: {recall_results['class_recall'][k]:.4f}")
    
    return recall_results

def evaluate_multi_viewgroup_retrieval(model, feature_db, dataset, device, transform):
    """
    评估多视点组输入检索object
    
    Args:
        model: 评估模型
        feature_db: 特征数据库
        dataset: 验证数据集
        device: 计算设备
        transform: 图像变换
    
    Returns:
        recall_results: 召回率结果
    """
    print("\n=== 多视点组输入检索object ===")
    
    # 构建查询集
    queries = {}
    for i in tqdm(range(len(dataset)), desc="生成查询"):
        data = dataset[i]
        obj_id = data['obj_id']
        images = data['images']  # [num_views, num_images_per_view, C, H, W]
        
        # 使用所有视点组
        viewgroup_images_list = [images[j] for j in range(images.shape[0])]
        
        # 生成对象特征
        obj_feat = multi_viewgroup_retrieval(model, viewgroup_images_list, device, transform)
        queries[obj_id] = obj_feat
    
    # 获取数据库
    database = feature_db.get_object_features()
    class_map = feature_db.get_class_map()
    
    # 计算召回率
    recall_results = compute_recall(queries, database, class_map)
    
    # 打印结果
    print("多视点组检索结果:")
    for k in sorted(recall_results['object_recall'].keys()):
        print(f"  Top-{k} Object Recall: {recall_results['object_recall'][k]:.4f}")
        print(f"  Top-{k} Class Recall: {recall_results['class_recall'][k]:.4f}")
    
    return recall_results

def evaluate_viewgroup_cross_retrieval(model, feature_db, dataset, device, transform):
    """
    评估视点组之间互检索
    
    Args:
        model: 评估模型
        feature_db: 特征数据库
        dataset: 验证数据集
        device: 计算设备
        transform: 图像变换
    
    Returns:
        recall_results: 召回率结果
    """
    print("\n=== 视点组之间互检索 ===")
    
    # 构建查询集
    queries = {}
    for i in tqdm(range(len(dataset)), desc="生成查询"):
        data = dataset[i]
        obj_id = data['obj_id']
        images = data['images']  # [num_views, num_images_per_view, C, H, W]
        
        # 使用所有视点组
        for j in range(images.shape[0]):
            view_key = f"{obj_id}_view_{j}"
            viewgroup_images = images[j]
            
            # 生成视图特征
            view_feat = viewgroup_cross_retrieval(model, viewgroup_images, device, transform)
            queries[view_key] = view_feat
    
    # 获取数据库
    database = feature_db.get_view_features()
    class_map = feature_db.get_class_map()
    
    # 计算召回率
    recall_results = compute_recall(queries, database, class_map)
    
    # 打印结果
    print("视点组互检索结果:")
    for k in sorted(recall_results['object_recall'].keys()):
        print(f"  Top-{k} View Recall: {recall_results['object_recall'][k]:.4f}")
        print(f"  Top-{k} Class Recall: {recall_results['class_recall'][k]:.4f}")
    
    return recall_results

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多视图检索系统 - 验证模块')
    
    # 数据参数
    parser.add_argument('--root_dir', type=str, default='/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0',
                        help='数据集根目录')
    parser.add_argument('--split', type=str, default='val',
                        help='数据集分割 (train/val/test)')
    parser.add_argument('--num_views', type=int, default=3,
                        help='训练时设置的视点组数量')
    parser.add_argument('--num_images_per_view', type=int, default=5,
                        help='每个视点组的图像数量')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像尺寸')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练得到的模型pth文件路径')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度大小')
    parser.add_argument('--num_classes', type=int, default=40,
                        help='类别数量')
    
    # 特征数据库参数
    parser.add_argument('--feature_db_path', type=str, required=True,
                        help='特征数据库路径')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='结果输出目录')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args, device)
    
    # 打开特征数据库
    feature_db = FeatureDatabase(args.feature_db_path)
    feature_db.open()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建验证数据集
    dataset = MultiViewDataset(
        root_dir=args.root_dir,
        transform=transform,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view,
        split=args.split
    )
    
    print(f"验证数据集加载完成: {len(dataset)}个样本")
    
    # 执行验证
    all_results = {}
    
    # 1. 单视点组输入检索object
    single_view_results = evaluate_single_viewgroup_retrieval(
        model, feature_db, dataset, args.num_views, device, transform
    )
    all_results['single_view_retrieval'] = single_view_results
    
    # 2. 多视点组输入检索object
    multi_view_results = evaluate_multi_viewgroup_retrieval(
        model, feature_db, dataset, device, transform
    )
    all_results['multi_view_retrieval'] = multi_view_results
    
    # 3. 视点组之间互检索
    view_cross_results = evaluate_viewgroup_cross_retrieval(
        model, feature_db, dataset, device, transform
    )
    all_results['view_cross_retrieval'] = view_cross_results
    
    # 保存结果
    import json
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # 将所有张量转换为可序列化的类型
        def serialize_results(results):
            if isinstance(results, dict):
                return {k: serialize_results(v) for k, v in results.items()}
            elif isinstance(results, float):
                return results
            elif hasattr(results, 'item'):
                return results.item()
            else:
                return results
        
        json.dump(serialize_results(all_results), f, indent=2, ensure_ascii=False)
    
    print(f"\n验证结果已保存至: {results_path}")
    
    # 关闭特征数据库
    feature_db.close()
    
    print("\n=== 验证完成 ===")
    
    # 打印总体结果
    print("\n总体验证结果:")
    print("单视点组检索:")
    for k in sorted(all_results['single_view_retrieval']['object_recall'].keys()):
        print(f"  Top-{k} Object Recall: {all_results['single_view_retrieval']['object_recall'][k]:.4f}")
    
    print("多视点组检索:")
    for k in sorted(all_results['multi_view_retrieval']['object_recall'].keys()):
        print(f"  Top-{k} Object Recall: {all_results['multi_view_retrieval']['object_recall'][k]:.4f}")
    
    print("视点组互检索:")
    for k in sorted(all_results['view_cross_retrieval']['object_recall'].keys()):
        print(f"  Top-{k} View Recall: {all_results['view_cross_retrieval']['object_recall'][k]:.4f}")

if __name__ == "__main__":
    main()
