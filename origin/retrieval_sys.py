#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch
import h5py
from tqdm import tqdm

class RetrievalSystem:
    def __init__(self, feature_db=None, dataset=None, device='cpu', model=None, class_labels=None):
        """
        初始化检索系统，支持三种检索情况：
        1. 单/多视图检索object，统计相同obj_id的object的召回率
        2. 单/多视图检索同obj其余视图，统计相同obj_id下其余view的召回率
        3. 上述两种检索均统计同class级别样本的召回率
        """
        self.device = device
        self.model = model
        self.class_labels = class_labels or {}
        if self.model:
            self.model = self.model.to(device)
        
        # 统一使用 self.features 存储特征
        self.features = {}
        
        # 存储对象ID到类标签的映射
        self.obj_to_class = {}
        
        # 存储对象ID到视图索引的映射（用于视图检索）
        self.obj_to_view_indices = {}
        
        if feature_db:
            # 从数据库加载所有特征到 self.features
            with h5py.File(feature_db, 'r') as db:
                for obj_id in db.keys():
                    obj_group = db[obj_id]
                    features_dict = {
                        'view_features': obj_group['view_features'][:],  # 保持为NumPy数组
                        'mid_features': obj_group['mid_features'][:].reshape(-1),  # 确保是一维数组
                        'obj_feat': obj_group['obj_feat'][:]
                    }
                    # 如果存在global_features，也添加到字典中
                    if 'global_features' in obj_group:
                        features_dict['global_features'] = obj_group['global_features'][:].reshape(-1)
                    # 如果存在class_label，也添加到字典中
                    if 'class_label' in obj_group:
                        features_dict['class_label'] = obj_group['class_label'][()]
                        self.obj_to_class[obj_id] = features_dict['class_label']
                    self.features[obj_id] = features_dict
        elif dataset and model:
            # 原始方式，使用数据集
            self.dataset = dataset
            # 构建特征并存储到 self.features
            self.features = self.build_database(dataset)
        else:
            raise ValueError("必须提供feature_db或dataset和模型")
    
    def build_database(self, dataset, batch_size=16, num_workers=4):
        """构建特征数据库
        
        Args:
            dataset: 数据集对象
            batch_size: 批处理大小
            num_workers: 数据加载线程数
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法构建数据库")
        
        # 初始化数据库
        self.features = {}
        self.obj_to_class = {}
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda batch: {
                'images': torch.stack([item['images'] for item in batch]),
                'obj_ids': [item['obj_id'] for item in batch],
                'labels': torch.tensor([item['label'] for item in batch]) if 'label' in batch[0] else None
            }
        )
        
        # 用于存储前几个对象的特征，以便比较
        sample_features = []
        sample_obj_ids = []
        
        # 提取特征
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['images'].to(self.device)
                obj_ids = batch['obj_ids']
                labels = batch.get('labels', None)
                
                # 处理每个视点
                num_views = images.shape[1]
                if num_views < 3:
                    # 如果少于3个视点，复制已有视点以满足要求
                    images = images.repeat(1, 3, 1, 1, 1, 1)[:, :3, :, :, :, :]
                
                view1 = images[:, 0]
                view2 = images[:, 1]
                view3 = images[:, 2]
                
                # 提取特征
                outputs = self.model(view1, view2, view3)
                
                # 添加详细调试信息检查特征
                if batch_idx == 0:  # 只在第一批数据中打印调试信息
                    print("\n===== 特征调试信息 =====")
                    print(f"批量 {batch_idx} 包含 {len(obj_ids)} 个对象")
                    
                    obj_feat_batch = outputs['obj_feat']
                    print(f"调试信息(特征构建) - 第一批特征统计:")
                    print(f"  obj_feat形状: {obj_feat_batch.shape}")
                    print(f"  obj_feat均值: {obj_feat_batch.mean().item():.6f}")
                    print(f"  obj_feat标准差: {obj_feat_batch.std().item():.6f}")
                    print(f"  第一个特征向量前5个值: {obj_feat_batch[0, :5].tolist()}")
                    
                    # 打印前3个对象的特征详细信息
                    for i in range(min(3, len(obj_ids))):
                        obj_id = obj_ids[i]
                        feat = outputs['obj_feat'][i].cpu().numpy().reshape(-1)
                        sample_features.append(feat)
                        sample_obj_ids.append(obj_id)
                        
                        print(f"\n对象 {obj_id} 特征信息:")
                        print(f"  形状: {feat.shape}")
                        print(f"  范数: {np.linalg.norm(feat):.6f}")
                        print(f"  均值: {np.mean(feat):.6f}")
                        print(f"  标准差: {np.std(feat):.6f}")
                        print(f"  前10个值: {feat[:10]}")
                    
                    # 检查特征之间的差异
                    if len(sample_features) >= 2:
                        print("\n特征差异性检查:")
                        for i in range(len(sample_features)):
                            for j in range(i+1, len(sample_features)):
                                diff = np.linalg.norm(sample_features[i] - sample_features[j])
                                # 计算余弦相似度
                                norm_i = np.linalg.norm(sample_features[i])
                                norm_j = np.linalg.norm(sample_features[j])
                                if norm_i > 1e-8 and norm_j > 1e-8:  # 避免除零错误
                                    cos_sim = np.dot(sample_features[i], sample_features[j]) / (norm_i * norm_j)
                                    print(f"  对象 {sample_obj_ids[i]} 和 {sample_obj_ids[j]} 特征差异: {diff:.6f}")
                                    print(f"  对象 {sample_obj_ids[i]} 和 {sample_obj_ids[j]} 余弦相似度: {cos_sim:.6f}")
                                    
                                    if diff < 1e-3:
                                        print(f"  警告: 对象 {sample_obj_ids[i]} 和 {sample_obj_ids[j]} 的特征几乎相同！")
                    
                    # 检查前两个特征向量的差异
                    if obj_feat_batch.shape[0] >= 2:
                        feat_diff = torch.norm(obj_feat_batch[0] - obj_feat_batch[1]).item()
                        print(f"  前两个特征向量差异: {feat_diff:.6f}")
                    
                    # 检查特征向量是否归一化
                    norms = torch.norm(obj_feat_batch, dim=1)
                    print(f"  特征向量范数范围: [{norms.min().item():.6f}, {norms.max().item():.6f}]")
                    
                    print("======================\n")
                
                # 保存特征
                for i, obj_id in enumerate(obj_ids):
                    # 获取对象特征
                    obj_feat = outputs['obj_feat'][i].cpu().numpy().reshape(-1)
                    
                    # 调试：检查特征向量是否全相同
                    if i > 0 and batch_idx == 0:
                        first_obj_id = obj_ids[0]
                        if first_obj_id in self.features:
                            prev_feat = self.features[first_obj_id]['obj_feat']
                            diff = np.linalg.norm(obj_feat - prev_feat)
                            if diff < 1e-3:  # 如果差异很小
                                print(f"警告：对象 {obj_id} 的特征向量与对象 {first_obj_id} 几乎相同！差异: {diff:.6f}")
                    
                    # 构建特征字典 - 确保所有特征都是一维数组
                    features_dict = {
                        'view_features': outputs['view_feats'][i].cpu().numpy(),
                        'mid_features': outputs['mid_feat'][i].cpu().numpy().reshape(-1),
                        'obj_feat': obj_feat
                    }
                    # 如果存在global_feat，也添加到字典中
                    if 'global_feat' in outputs:
                        features_dict['global_features'] = outputs['global_feat'][i].cpu().numpy().reshape(-1)
                    # 如果存在标签，也添加到字典中
                    if labels is not None:
                        features_dict['class_label'] = labels[i].item()
                        self.obj_to_class[obj_id] = features_dict['class_label']
                    
                    # 修复：使用self.features存储特征
                    self.features[obj_id] = features_dict
        
        # 打印数据库统计信息
        print(f"调试信息 - 特征数据库构建完成，包含 {len(self.features)} 个对象")
        if self.features:
            # 随机选择两个对象检查特征差异
            obj_keys = list(self.features.keys())
            if len(obj_keys) >= 2:
                feat1 = self.features[obj_keys[0]]['obj_feat']
                feat2 = self.features[obj_keys[1]]['obj_feat']
                diff = np.linalg.norm(feat1 - feat2)
                
                # 计算余弦相似度
                norm1 = np.linalg.norm(feat1)
                norm2 = np.linalg.norm(feat2)
                if norm1 > 1e-8 and norm2 > 1e-8:  # 避免除零错误
                    cos_sim = np.dot(feat1, feat2) / (norm1 * norm2)
                    print(f"调试信息 - 随机两个对象特征向量差异: {diff:.6f}")
                    print(f"调试信息 - 随机两个对象余弦相似度: {cos_sim:.6f}")
        
        return self.features
    
    def query_by_single_view(self, view_images, topk=5, return_class_stats=False):
        """使用单视图检索object
        
        Args:
            view_images: 单视图图像张量
            topk: 返回前k个结果
            return_class_stats: 是否返回类别统计信息
            
        Returns:
            检索结果列表，每个元素为(obj_id, similarity)元组
            如果return_class_stats=True，还返回字典包含obj_id和class召回率
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法执行查询")
        
        with torch.no_grad():
            # 处理查询图像
            view_images = view_images.unsqueeze(0).to(self.device)  # 添加批次维度
            
            # 提取特征 (使用空视点作为占位符)
            outputs = self.model(view_images, view_images, view_images)
            
            query_feat = outputs['obj_feat'][0].cpu().numpy()
        
        # 执行检索
        results = self._retrieve_objects(query_feat, topk)
        
        if return_class_stats and results:
            # 假设第一个结果是查询对象本身（如果存在）
            query_obj_id = results[0][0] if results else None
            if query_obj_id and query_obj_id in self.obj_to_class:
                query_class = self.obj_to_class[query_obj_id]
                # 计算obj_id召回率（假设我们知道查询的obj_id）
                obj_recall = 1.0 if any(res[0] == query_obj_id for res in results) else 0.0
                # 计算class召回率
                class_recall = 1.0 if any(self.obj_to_class.get(res[0], -1) == query_class for res in results) else 0.0
                return results, {'obj_recall': obj_recall, 'class_recall': class_recall}
        
        return results
    
    def query_by_multiple_views(self, view1, view2, view3, topk=5, return_class_stats=False):
        """使用多视图检索object
        
        Args:
            view1, view2, view3: 三个视点的图像张量
            topk: 返回前k个结果
            return_class_stats: 是否返回类别统计信息
            
        Returns:
            检索结果列表，每个元素为(obj_id, similarity)元组
            如果return_class_stats=True，还返回字典包含obj_id和class召回率
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法执行查询")
        
        with torch.no_grad():
            # 处理查询图像
            view1 = view1.unsqueeze(0).to(self.device)
            view2 = view2.unsqueeze(0).to(self.device)
            view3 = view3.unsqueeze(0).to(self.device)
            
            # 提取特征
            outputs = self.model(view1, view2, view3)
            query_feat = outputs['obj_feat'][0].cpu().numpy()
        
        # 执行检索
        results = self._retrieve_objects(query_feat, topk)
        
        if return_class_stats and results:
            # 假设第一个结果是查询对象本身（如果存在）
            query_obj_id = results[0][0] if results else None
            if query_obj_id and query_obj_id in self.obj_to_class:
                query_class = self.obj_to_class[query_obj_id]
                # 计算obj_id召回率（假设我们知道查询的obj_id）
                obj_recall = 1.0 if any(res[0] == query_obj_id for res in results) else 0.0
                # 计算class召回率
                class_recall = 1.0 if any(self.obj_to_class.get(res[0], -1) == query_class for res in results) else 0.0
                return results, {'obj_recall': obj_recall, 'class_recall': class_recall}
        
        return results
    
    def query_same_obj_views(self, query_obj_id, query_feat=None, exclude_view_index=None, topk=5):
        """使用对象的obj_feat检索相同对象的其余视图
        
        Args:
            query_obj_id: 查询的对象ID
            query_feat: 查询特征（如果提供）
            exclude_view_index: 要排除的视图索引（如果要检索除当前视图外的其他视图）
            topk: 返回前k个结果
            
        Returns:
            检索结果列表，每个元素为(obj_id, view_index, similarity)元组
        """
        if query_obj_id not in self.features:
            raise ValueError(f"对象ID {query_obj_id} 不在数据库中")
        
        # 如果没有提供查询特征，使用对象的obj_feat
        if query_feat is None:
            query_feat = self.features[query_obj_id]['obj_feat']
        
        # 执行视图检索
        return self._retrieve_views(query_feat, exclude_obj_id=query_obj_id, 
                                   exclude_view_index=exclude_view_index, topk=topk)
    
    def query_by_view_for_same_obj_views(self, view_images, obj_id=None, exclude_view_index=None, topk=5):
        """使用单视图检索相同对象的其余视图
        
        Args:
            view_images: 单视图图像张量
            obj_id: 对象ID（如果已知）
            exclude_view_index: 要排除的视图索引（如果要检索除当前视图外的其他视图）
            topk: 返回前k个结果
            
        Returns:
            检索结果列表，每个元素为(obj_id, view_index, similarity)元组
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法执行查询")
        
        with torch.no_grad():
            # 处理查询图像
            view_images = view_images.unsqueeze(0).to(self.device)  # 添加批次维度
            
            # 提取特征 (使用空视点作为占位符)
            outputs = self.model(view_images, view_images, view_images)
            query_feat = outputs['obj_feat'][0].cpu().numpy()
        
        # 如果提供了对象ID，直接使用
        if obj_id:
            return self.query_same_obj_views(obj_id, query_feat=query_feat, 
                                           exclude_view_index=exclude_view_index, topk=topk)
        
        # 否则，首先找到匹配的对象
        obj_results = self._retrieve_objects(query_feat, topk=1)
        if not obj_results:
            return []
        
        # 使用匹配对象的特征检索视图
        matched_obj_id = obj_results[0][0]
        return self._retrieve_views(query_feat, exclude_obj_id=matched_obj_id, 
                                   exclude_view_index=exclude_view_index, topk=topk)
    
    def _retrieve_objects(self, query_feat, topk=5):
        """检索对象
        
        Args:
            query_feat: 查询特征
            topk: 返回前k个结果
            
        Returns:
            检索结果列表，每个元素为(obj_id, similarity)元组
        """
        similarities = []
        epsilon = 1e-10  # 添加小值避免除零错误
        
        # 添加详细调试信息
        print("\n===== 检索调试信息 =====")
        query_norm = np.linalg.norm(query_feat)
        print(f"查询特征形状: {query_feat.shape}")
        print(f"查询特征范数: {query_norm:.6f}")
        print(f"查询特征完整向量: {query_feat}")
        print(f"查询特征均值: {np.mean(query_feat):.6f}")
        print(f"查询特征标准差: {np.std(query_feat):.6f}")
        
        # 打印前3个数据库对象的特征信息
        db_items = list(self.features.items())[:3]
        for obj_id, features in db_items:
            obj_feat = features['obj_feat']
            obj_norm = np.linalg.norm(obj_feat)
            print(f"\n数据库对象 {obj_id} 特征信息:")
            print(f"  特征范数: {obj_norm:.6f}")
            print(f"  特征完整向量: {obj_feat}")
            print(f"  特征均值: {np.mean(obj_feat):.6f}")
            print(f"  特征标准差: {np.std(obj_feat):.6f}")
            
            # 计算点积和余弦相似度
            dot_product = np.dot(query_feat, obj_feat)
            if query_norm > 1e-8 and obj_norm > 1e-8:
                cos_sim = dot_product / (query_norm * obj_norm)
                print(f"  与查询特征点积: {dot_product:.6f}")
                print(f"  与查询特征余弦相似度: {cos_sim:.6f}")
            else:
                print(f"  警告: 查询特征或对象特征范数接近零")
        
        if query_norm < epsilon:
            # 查询特征无效，所有相似度为0
            for obj_id in self.features.keys():
                similarities.append((obj_id, 0.0))
        else:
            # 重新归一化查询特征以确保正确性
            normalized_query = query_feat / query_norm
            
            # 计算与每个对象的相似度
            for obj_id, features in self.features.items():
                obj_feat = features['obj_feat']
                feat_norm = np.linalg.norm(obj_feat)
                
                # 检查特征有效性
                if feat_norm < epsilon:
                    sim = 0.0
                else:
                    # 归一化对象特征
                    normalized_obj = obj_feat / feat_norm
                    # 计算点积
                    sim = np.dot(normalized_query, normalized_obj)
                    
                    # 确保相似度在合理范围内
                    sim = max(-1.0, min(1.0, sim))
                    
                similarities.append((obj_id, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 打印检索结果
        print(f"\n前5个检索结果:")
        for i, (obj_id, sim) in enumerate(similarities[:5]):
            print(f"  {i+1}. {obj_id}: {sim:.6f}")
        print("======================\n")
        
        # 返回topk结果
        return similarities[:topk]
    
    def _retrieve_views(self, query_feat, exclude_obj_id=None, exclude_view_index=None, topk=5):
        """检索视图
        
        Args:
            query_feat: 查询特征
            exclude_obj_id: 要排除的对象ID
            exclude_view_index: 要排除的视图索引（仅在指定exclude_obj_id时有效）
            topk: 返回前k个结果
            
        Returns:
            检索结果列表，每个元素为(obj_id, view_index, similarity)元组
        """
        similarities = []
        epsilon = 1e-10  # 添加小值避免除零错误
        
        # 添加调试信息检查查询特征
        query_norm = np.linalg.norm(query_feat)
        # 打印查询特征的范数，用于调试
        print(f"调试信息(View) - 查询特征范数: {query_norm:.6f}")
        
        # 如果查询特征无效，所有相似度为0
        if query_norm < epsilon:
            # 处理所有对象和视图
            for obj_id, features in self.features.items():
                if exclude_obj_id and obj_id == exclude_obj_id:
                    if exclude_view_index is not None:
                        for i, _ in enumerate(features['view_features']):
                            if i != exclude_view_index:
                                similarities.append((obj_id, i, 0.0))
                else:
                    for i in range(len(features['view_features'])):
                        similarities.append((obj_id, i, 0.0))
        else:
            # 重新归一化查询特征以确保正确性
            normalized_query = query_feat / query_norm
            
            # 计算与每个视图的相似度
            for obj_id, features in self.features.items():
                # 跳过要排除的对象
                if exclude_obj_id and obj_id == exclude_obj_id:
                    # 如果指定了要排除的视图索引，则只排除该视图
                    if exclude_view_index is not None:
                        for i, view_feat in enumerate(features['view_features']):
                            if i != exclude_view_index:
                                feat_norm = np.linalg.norm(view_feat)
                                if feat_norm < epsilon:
                                    sim = 0.0
                                else:
                                    # 归一化视图特征
                                    normalized_view = view_feat / feat_norm
                                    # 计算点积
                                    sim = np.dot(normalized_query, normalized_view)
                                    sim = max(-1.0, min(1.0, sim))
                                similarities.append((obj_id, i, sim))
                # 否则处理所有视图
                else:
                    # 对所有视图计算相似度
                    for i, view_feat in enumerate(features['view_features']):
                        feat_norm = np.linalg.norm(view_feat)
                        if feat_norm < epsilon:
                            sim = 0.0
                        else:
                            # 归一化视图特征
                            normalized_view = view_feat / feat_norm
                            # 计算点积
                            sim = np.dot(normalized_query, normalized_view)
                            sim = max(-1.0, min(1.0, sim))
                        similarities.append((obj_id, i, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # 打印前3个结果用于调试
        if similarities:
            print(f"调试信息(View) - 前3个检索结果: {similarities[:3]}")
        
        # 返回topk结果
        return similarities[:topk]
    

    
    def get_view_features(self, obj_id, view_index):
        """获取特定视点特征"""
        if obj_id not in self.features:
            raise ValueError(f"对象ID {obj_id} 不在数据库中")
        if view_index >= len(self.features[obj_id]['view_features']):
            raise ValueError(f"视图索引 {view_index} 超出范围")
        return self.features[obj_id]['view_features'][view_index]

    def evaluate_object_retrieval(self, dataset, topk_list=[1, 5, 10], use_single_view=True, use_multiple_views=True):
        """评估对象检索性能
        
        Args:
            dataset: 数据集对象
            topk_list: 要评估的top-k值列表
            use_single_view: 是否评估单视图检索
            use_multiple_views: 是否评估多视图检索
            
        Returns:
            评估结果字典
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法执行评估")
        
        # 初始化结果存储
        results = {
            'single_view': {},
            'multiple_views': {},
            'overall': {}
        }
        
        # 为每个topk值初始化统计
        for topk in topk_list:
            results['single_view'][topk] = {
                'obj_recall': 0.0,
                'class_recall': 0.0,
                'total_queries': 0
            }
            results['multiple_views'][topk] = {
                'obj_recall': 0.0,
                'class_recall': 0.0,
                'total_queries': 0
            }
        
        # 遍历数据集进行查询
        dataloader = DataLoader(
            dataset, 
            batch_size=1,  # 逐个查询
            shuffle=False,
            num_workers=0
        )
        
        print("开始评估对象检索性能...")
        for i, batch in enumerate(tqdm(dataloader, desc="评估进度")):
            obj_id = batch['obj_id'][0]
            images = batch['images'][0]  # [num_views, num_images_per_view, C, H, W]
            
            # 单视图检索评估
            if use_single_view:
                for view_idx in range(len(images)):
                    view_images = images[view_idx]  # [num_images_per_view, C, H, W]
                    
                    # 使用单视图检索
                    retrieval_results = self.query_by_single_view(
                        view_images, 
                        topk=max(topk_list),
                        return_class_stats=True
                    )
                    
                    # 打印更详细的检索结果，显示具体的相似度值（保留6位小数）
                    if isinstance(retrieval_results, tuple):
                        results_list, _ = retrieval_results
                        formatted_results = [(res[0], f"{res[1]:.6f}") for res in results_list[:10]]
                        print(f'单视图检索结果: query {obj_id} : {formatted_results}')
                        
                        # 调试：检查数据库中特征向量的差异
                        if len(results_list) >= 2:
                            # 检查query_id是否在数据库中
                            if obj_id in self.features:
                                query_feat = self.features[obj_id]['obj_feat']
                                print(f"调试信息 - 查询对象 {obj_id} 特征范数: {np.linalg.norm(query_feat):.6f}")
                                print(f"调试信息 - 查询特征前5个值: {query_feat[:5]}")
                            
                            # 检查前两个检索结果的特征
                            for j in range(min(2, len(results_list))):
                                retrieved_id, sim = results_list[j]
                                if retrieved_id in self.features:
                                    retrieved_feat = self.features[retrieved_id]['obj_feat']
                                    print(f"调试信息 - 检索结果 {j+1} ({retrieved_id}) 特征范数: {np.linalg.norm(retrieved_feat):.6f}")
                                    print(f"调试信息 - 检索特征前5个值: {retrieved_feat[:5]}")
                            
                            # 计算前两个检索结果的特征向量差异
                            if results_list[0][0] in self.features and results_list[1][0] in self.features:
                                feat1 = self.features[results_list[0][0]]['obj_feat']
                                feat2 = self.features[results_list[1][0]]['obj_feat']
                                feat_diff = np.linalg.norm(feat1 - feat2)
                                print(f"调试信息 - 前两个检索结果特征差异: {feat_diff:.6f}")
                    else:
                        print(f'单视图检索结果: query {obj_id} : {retrieval_results}')
                    
                    if isinstance(retrieval_results, tuple):
                        results_list, stats = retrieval_results
                        
                        # 输出当前object的检索结果
                        print(f"\nObject {obj_id} - 视图 {view_idx} 检索结果:")
                        print(f"查询对象: {obj_id}")
                        print(f"Top-{max(topk_list)} 检索结果:")
                        for j, (retrieved_obj_id, similarity) in enumerate(results_list[:10]):  # 只显示前10个结果
                            retrieved_class = self.obj_to_class.get(retrieved_obj_id, "未知")
                            is_correct = retrieved_obj_id == obj_id
                            print(f"  {j+1}. 对象: {retrieved_obj_id}, 类别: {retrieved_class}, 相似度: {similarity:.4f}{' ✓' if is_correct else ''}")
                        
                        # 更新每个topk的统计
                        for topk in topk_list:
                            topk_results = results_list[:topk]
                            
                            # 检查obj_id是否在topk结果中
                            obj_found = any(res[0] == obj_id for res in topk_results)
                            
                            # 检查类别是否在topk结果中
                            if obj_id in self.obj_to_class:
                                query_class = self.obj_to_class[obj_id]
                                class_found = any(self.obj_to_class.get(res[0], -1) == query_class for res in topk_results)
                            else:
                                class_found = False
                            
                            if obj_found:
                                results['single_view'][topk]['obj_recall'] += 1
                            if class_found:
                                results['single_view'][topk]['class_recall'] += 1
                            results['single_view'][topk]['total_queries'] += 1
                            
                            # 输出当前查询的Top-K结果
                            print(f"  Top-{topk}: 对象召回: {'✓' if obj_found else '✗'}, 类别召回: {'✓' if class_found else '✗'}")
            
            # 多视图检索评估
            if use_multiple_views and len(images) >= 3:
                # 使用前三个视图进行多视图检索
                view1 = images[0]
                view2 = images[1] 
                view3 = images[2]
                
                retrieval_results = self.query_by_multiple_views(
                    view1, view2, view3,
                    topk=max(topk_list),
                    return_class_stats=True
                )
                
                if isinstance(retrieval_results, tuple):
                    results_list, stats = retrieval_results
                    
                    # 输出当前object的多视图检索结果
                    print(f"\nObject {obj_id} - 多视图检索结果:")
                    print(f"查询对象: {obj_id}")
                    print(f"Top-{max(topk_list)} 检索结果:")
                    for j, (retrieved_obj_id, similarity) in enumerate(results_list[:10]):  # 只显示前10个结果
                        retrieved_class = self.obj_to_class.get(retrieved_obj_id, "未知")
                        is_correct = retrieved_obj_id == obj_id
                        print(f"  {j+1}. 对象: {retrieved_obj_id}, 类别: {retrieved_class}, 相似度: {similarity:.4f}{' ✓' if is_correct else ''}")
                    
                    # 更新每个topk的统计
                    for topk in topk_list:
                        topk_results = results_list[:topk]
                        
                        # 检查obj_id是否在topk结果中
                        obj_found = any(res[0] == obj_id for res in topk_results)
                        
                        # 检查类别是否在topk结果中
                        if obj_id in self.obj_to_class:
                            query_class = self.obj_to_class[obj_id]
                            class_found = any(self.obj_to_class.get(res[0], -1) == query_class for res in topk_results)
                        else:
                            class_found = False
                        
                        if obj_found:
                            results['multiple_views'][topk]['obj_recall'] += 1
                        if class_found:
                            results['multiple_views'][topk]['class_recall'] += 1
                        results['multiple_views'][topk]['total_queries'] += 1
                        
                        # 输出当前查询的Top-K结果
                        print(f"  Top-{topk}: 对象召回: {'✓' if obj_found else '✗'}, 类别召回: {'✓' if class_found else '✗'}")
        
        # 计算平均召回率
        for mode in ['single_view', 'multiple_views']:
            for topk in topk_list:
                if results[mode][topk]['total_queries'] > 0:
                    results[mode][topk]['obj_recall'] /= results[mode][topk]['total_queries']
                    results[mode][topk]['class_recall'] /= results[mode][topk]['total_queries']
        
        # 计算总体结果
        for topk in topk_list:
            total_obj_recall = 0
            total_class_recall = 0
            total_queries = 0
            
            for mode in ['single_view', 'multiple_views']:
                if results[mode][topk]['total_queries'] > 0:
                    total_obj_recall += results[mode][topk]['obj_recall'] * results[mode][topk]['total_queries']
                    total_class_recall += results[mode][topk]['class_recall'] * results[mode][topk]['total_queries']
                    total_queries += results[mode][topk]['total_queries']
            
            if total_queries > 0:
                results['overall'][topk] = {
                    'obj_recall': total_obj_recall / total_queries,
                    'class_recall': total_class_recall / total_queries,
                    'total_queries': total_queries
                }
        
        return results

    def evaluate_view_retrieval(self, dataset, topk_list=[1, 5, 10]):
        """评估视图检索性能（单视点组间互检索）
        
        Args:
            dataset: 数据集对象
            topk_list: 要评估的top-k值列表
            
        Returns:
            评估结果字典
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法执行评估")
        
        # 初始化结果存储
        results = {}
        for topk in topk_list:
            results[topk] = {
                'view_recall': 0.0,
                'total_queries': 0
            }
        
        # 遍历数据集进行查询
        dataloader = DataLoader(
            dataset, 
            batch_size=1,  # 逐个查询
            shuffle=False,
            num_workers=0
        )
        
        print("开始评估视图检索性能...")
        for i, batch in enumerate(tqdm(dataloader, desc="评估进度")):
            obj_id = batch['obj_id'][0]
            images = batch['images'][0]  # [num_views, num_images_per_view, C, H, W]
            
            # 对每个视图进行查询
            for view_idx in range(len(images)):
                view_images = images[view_idx]  # [num_images_per_view, C, H, W]
                
                # 使用视图检索相同对象的其他视图
                retrieval_results = self.query_by_view_for_same_obj_views(
                    view_images, 
                    obj_id=obj_id,
                    exclude_view_index=view_idx,  # 排除当前视图
                    topk=max(topk_list)
                )
                
                # 更新每个topk的统计
                for topk in topk_list:
                    topk_results = retrieval_results[:topk]
                    
                    # 检查是否检索到相同对象的其他视图
                    view_found = any(res[0] == obj_id for res in topk_results)
                    
                    if view_found:
                        results[topk]['view_recall'] += 1
                    results[topk]['total_queries'] += 1
        
        # 计算平均召回率
        for topk in topk_list:
            if results[topk]['total_queries'] > 0:
                results[topk]['view_recall'] /= results[topk]['total_queries']
        
        return results