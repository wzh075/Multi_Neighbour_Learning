import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import json

class RetrievalSystem:
    def __init__(self, model=None, feature_db=None, tensorboard_dir=None):
        """初始化检索系统
        
        Args:
            model: 特征提取模型
            feature_db: 特征数据库字典，格式为 {id: {'obj_feat': 特征向量, 'class': 类别}}
            tensorboard_dir: TensorBoard日志目录（可选）
        """
        self.model = model
        self.features = feature_db if feature_db else {}
        self.obj_to_class = {}
        self.tensorboard_dir = tensorboard_dir
        self.writer = None
        
        # 如果提供了tensorboard_dir，则初始化writer
        if tensorboard_dir is not None:
            try:
                # 优先使用torch.utils.tensorboard
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=tensorboard_dir)
                print(f"TensorBoard SummaryWriter已初始化，日志目录: {tensorboard_dir}")
            except ImportError:
                try:
                    # 备选方案：使用tensorboardX
                    from tensorboardX import SummaryWriter
                    self.writer = SummaryWriter(log_dir=tensorboard_dir)
                    print(f"TensorboardX SummaryWriter已初始化，日志目录: {tensorboard_dir}")
                except ImportError:
                    print("警告: 无法导入tensorboard或tensorboardX，将不会记录到TensorBoard")
                    self.writer = None
        
        # 从特征数据库构建对象到类别的映射
        for obj_id, data in self.features.items():
            if 'class' in data:
                self.obj_to_class[obj_id] = data['class']
        
        print(f"检索系统初始化完成，包含 {len(self.features)} 个对象的特征")
    
    def build_database(self, dataset, batch_size=32, device='cuda'):
        """从数据集构建特征数据库
        
        Args:
            dataset: 数据集对象
            batch_size: 批量大小
            device: 设备（会被模型实际所在设备覆盖）
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法构建数据库")
        
        self.model.eval()
        self.model.to(device)  # 先将模型移动到指定设备
        # 获取模型实际所在的设备
        actual_device = next(self.model.parameters()).device
        print(f"模型运行在设备: {actual_device}")
        
        self.features = {}
        self.obj_to_class = {}
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        print("开始构建特征数据库...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="特征提取进度"):
                obj_ids = batch['obj_id']
                images = batch['images']  # [batch, num_views, num_images_per_view, C, H, W]
                classes = batch.get('class', [None] * len(obj_ids))
                
                # 计算每个对象的特征（对所有视图求平均）
                for i in range(len(obj_ids)):
                    obj_id = obj_ids[i]
                    obj_images = images[i]  # [num_views, num_images_per_view, C, H, W]
                    
                    # 确保有3个视图，并正确传递给模型
                    if obj_images.shape[0] >= 3:
                        # 提取前3个视图
                        view1 = obj_images[0].to(actual_device).unsqueeze(0)  # [1, num_images_per_view, C, H, W]
                        view2 = obj_images[1].to(actual_device).unsqueeze(0)
                        view3 = obj_images[2].to(actual_device).unsqueeze(0)
                        
                        # 调用模型获取特征
                        outputs = self.model(view1, view2, view3)
                        
                        # 使用对象级特征
                        obj_feat = outputs['obj_feat'].squeeze().cpu().numpy()
                    else:
                        # 如果视图不足3个，使用原来的方法作为备选
                        view_features = []
                        for view_images in obj_images:
                            # 尝试作为单个视图使用encode_single_view方法
                            try:
                                view_images = view_images.to(actual_device).unsqueeze(0)
                                view_feat = self.model.encode_single_view(view_images).squeeze().cpu().numpy()
                                view_features.append(view_feat)
                            except Exception as e:
                                print(f"警告: 处理视图时出错: {str(e)}")
                        
                        if view_features:
                            obj_feat = np.mean(np.array(view_features), axis=0)
                        else:
                            print(f"警告: 对象 {obj_id} 没有有效特征")
                            continue  # 跳过此对象
                    
                    # obj_feat已在前面计算完成，不再需要重新计算
                    
                    # 验证特征
                    if np.isnan(obj_feat).any() or np.isinf(obj_feat).any():
                        print(f"警告: 对象 {obj_id} 的特征包含 NaN 或 Inf 值")
                    
                    # 存储特征
                    self.features[obj_id] = {
                        'obj_feat': obj_feat
                    }
                    
                    # 存储类别信息
                    if classes[i] is not None:
                        self.features[obj_id]['class'] = classes[i]
                        self.obj_to_class[obj_id] = classes[i]
        
        print(f"特征数据库构建完成，包含 {len(self.features)} 个对象的特征")
    
    def query_by_single_view(self, image, topk=10, return_class_stats=False):
        """单视图检索
        
        Args:
            image: 单视图图像张量
            topk: 返回前k个结果
            return_class_stats: 是否返回类别统计信息
        
        Returns:
            检索结果列表或元组 (结果列表, 统计信息)
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法执行检索")
        
        self.model.eval()
        with torch.no_grad():
            # 确保输入数据和模型在同一设备上
            image = image.to(next(self.model.parameters()).device)
            # 提取查询图像特征 - 使用encode_single_view方法处理单个视图
            # 为了确保维度正确，unsqueeze(0)添加batch维度，再unsqueeze(0)添加num_images_per_view维度
            # 然后使用encode_single_view方法
            query_feat = self.model.encode_single_view(image.unsqueeze(0).unsqueeze(0)).cpu().numpy()
            
            # 计算与数据库中所有特征的相似度
            results = []
            for obj_id, data in self.features.items():
                obj_feat = data['obj_feat']
                # 计算余弦相似度
                sim = np.dot(query_feat, obj_feat) / (np.linalg.norm(query_feat) * np.linalg.norm(obj_feat) + 1e-8)
                results.append((obj_id, sim))
            
            # 按相似度降序排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前topk个结果
            if return_class_stats:
                # 收集类别统计信息
                class_counts = {}
                for obj_id, _ in results[:topk]:
                    if obj_id in self.obj_to_class:
                        cls = self.obj_to_class[obj_id]
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                return results[:topk], class_counts
            else:
                return results[:topk]
    
    def query_by_multiple_views(self, view1, view2, view3, topk=10, return_class_stats=False):
        """多视图检索
        
        Args:
            view1, view2, view3: 三个视图的图像张量
            topk: 返回前k个结果
            return_class_stats: 是否返回类别统计信息
        
        Returns:
            检索结果列表或元组 (结果列表, 统计信息)
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法执行检索")
        
        self.model.eval()
        with torch.no_grad():
            # 提取三个视图的特征 - 正确使用模型的forward方法
            # 添加batch和num_images_per_view维度
            view1 = view1.unsqueeze(0).unsqueeze(0).to(next(self.model.parameters()).device)
            view2 = view2.unsqueeze(0).unsqueeze(0).to(next(self.model.parameters()).device)
            view3 = view3.unsqueeze(0).unsqueeze(0).to(next(self.model.parameters()).device)
            
            # 使用模型的forward方法一次性处理三个视图
            outputs = self.model(view1, view2, view3)
            
            # 使用对象级特征作为查询特征
            query_feat = outputs['obj_feat'].squeeze().cpu().numpy()
            
            # 计算与数据库中所有特征的相似度
            results = []
            for obj_id, data in self.features.items():
                obj_feat = data['obj_feat']
                # 计算余弦相似度
                sim = np.dot(query_feat, obj_feat) / (np.linalg.norm(query_feat) * np.linalg.norm(obj_feat) + 1e-8)
                results.append((obj_id, sim))
            
            # 按相似度降序排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前topk个结果
            if return_class_stats:
                # 收集类别统计信息
                class_counts = {}
                for obj_id, _ in results[:topk]:
                    if obj_id in self.obj_to_class:
                        cls = self.obj_to_class[obj_id]
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                return results[:topk], class_counts
            else:
                return results[:topk]
    
    def query_same_obj_views(self, query_feat, obj_id, exclude_view_index=None, topk=10):
        """查询相同对象的其他视图
        
        Args:
            query_feat: 查询特征向量
            obj_id: 对象ID
            exclude_view_index: 要排除的视图索引
            topk: 返回前k个结果
        
        Returns:
            检索结果列表
        """
        results = []
        
        # 对于相同对象的所有特征（如果有视图级别的特征）
        if obj_id in self.features and 'view_feats' in self.features[obj_id]:
            view_feats = self.features[obj_id]['view_feats']
            
            for view_idx, view_feat in enumerate(view_feats):
                if exclude_view_index is not None and view_idx == exclude_view_index:
                    continue
                
                # 计算余弦相似度
                sim = np.dot(query_feat, view_feat) / (np.linalg.norm(query_feat) * np.linalg.norm(view_feat) + 1e-8)
                results.append((obj_id, sim, view_idx))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:topk]
    
    def query_by_view_for_same_obj_views(self, view_images, obj_id=None, exclude_view_index=None, topk=10):
        """使用视图检索相同对象的其他视图
        
        Args:
            view_images: 视图中的图像张量 [num_images_per_view, C, H, W]
            obj_id: 对象ID（可选）
            exclude_view_index: 要排除的视图索引
            topk: 返回前k个结果
        
        Returns:
            检索结果列表
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法执行检索")
        
        self.model.eval()
        with torch.no_grad():
            # 提取视图特征 - 使用encode_single_view方法处理单个视图
            # 为了确保维度正确，unsqueeze(0)添加batch维度
            view_images = view_images.to(next(self.model.parameters()).device).unsqueeze(0)
            query_feat = self.model.encode_single_view(view_images).cpu().numpy()
            
            return self.query_same_obj_views(query_feat, obj_id, exclude_view_index, topk)
    
    def _retrieve_objects(self, query_feat, topk=10, debug=False):
        """对象检索核心方法
        
        Args:
            query_feat: 查询特征向量
            topk: 返回前k个结果
            debug: 是否打印调试信息
        
        Returns:
            检索结果列表
        """
        results = []
        
        # 验证查询特征
        if np.isnan(query_feat).any() or np.isinf(query_feat).any():
            print("警告: 查询特征包含 NaN 或 Inf 值")
            return results
        
        if debug:
            print(f"调试信息 - 查询特征范数: {np.linalg.norm(query_feat):.6f}")
            print(f"调试信息 - 查询特征前5个值: {query_feat[:5]}")
        
        # 计算与数据库中所有特征的相似度
        for obj_id, data in self.features.items():
            obj_feat = data['obj_feat']
            
            # 验证对象特征
            if np.isnan(obj_feat).any() or np.isinf(obj_feat).any():
                continue
            
            # 计算余弦相似度
            sim = np.dot(query_feat, obj_feat) / (np.linalg.norm(query_feat) * np.linalg.norm(obj_feat) + 1e-8)
            results.append((obj_id, sim))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 调试信息
        if debug and results:
            for j in range(min(2, len(results))):
                retrieved_id, sim = results[j]
                if retrieved_id in self.features:
                    retrieved_feat = self.features[retrieved_id]['obj_feat']
                    print(f"调试信息 - 检索结果 {j+1} ({retrieved_id}) 特征范数: {np.linalg.norm(retrieved_feat):.6f}")
                    print(f"调试信息 - 检索特征前5个值: {retrieved_feat[:5]}")
            
            if len(results) >= 2:
                feat1 = self.features[results[0][0]]['obj_feat']
                feat2 = self.features[results[1][0]]['obj_feat']
                feat_diff = np.linalg.norm(feat1 - feat2)
                print(f"调试信息 - 前两个检索结果特征差异: {feat_diff:.6f}")
        
        return results[:topk]
    
    def _retrieve_views(self, query_feat, obj_id=None, topk=10):
        """视图检索核心方法
        
        Args:
            query_feat: 查询特征向量
            obj_id: 对象ID（可选）
            topk: 返回前k个结果
        
        Returns:
            检索结果列表
        """
        results = []
        
        # 对于指定对象或所有对象的视图特征
        target_objs = [obj_id] if obj_id else self.features.keys()
        
        for obj in target_objs:
            if obj in self.features and 'view_feats' in self.features[obj]:
                view_feats = self.features[obj]['view_feats']
                
                for view_idx, view_feat in enumerate(view_feats):
                    # 计算余弦相似度
                    sim = np.dot(query_feat, view_feat) / (np.linalg.norm(query_feat) * np.linalg.norm(view_feat) + 1e-8)
                    results.append((obj, sim, view_idx))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:topk]
    
    def get_view_features(self, dataset):
        """获取数据集所有视图的特征
        
        Args:
            dataset: 数据集对象
        
        Returns:
            视图特征字典
        """
        if not self.model:
            raise RuntimeError("未提供模型，无法获取视图特征")
        
        view_features = {}
        
        dataloader = DataLoader(
            dataset, 
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="获取视图特征"):
                obj_id = batch['obj_id'][0]
                images = batch['images'][0]  # [num_views, num_images_per_view, C, H, W]
                
                obj_view_feats = []
                # 获取模型所在设备
                device = next(self.model.parameters()).device
                for view_images in images:
                    # 提取视图特征 - 使用encode_single_view方法处理单个视图
                    # 为了确保维度正确，unsqueeze(0)添加batch维度
                    # 确保输入数据和模型在同一设备上
                    view_images = view_images.to(device).unsqueeze(0)
                    view_feat = self.model.encode_single_view(view_images).cpu().numpy()
                    obj_view_feats.append(view_feat)
                
                view_features[obj_id] = obj_view_feats
        
        return view_features
    
    def evaluate_object_retrieval(self, dataset, topk_list=[1, 5, 10], use_multiple_views=False, debug=False):
        """评估对象检索性能
        
        Args:
            dataset: 数据集对象
            topk_list: 要评估的top-k值列表
            use_multiple_views: 是否使用多视图检索
            debug: 是否打印调试信息
        
        Returns:
            评估结果字典
        """
        eval_step = 0
        # 初始化结果存储
        results = {
            'single_view': {},
            'multiple_views': {},
            'overall': {}
        }
        
        # 初始化每个topk的结果
        for topk in topk_list:
            results['single_view'][topk] = {
                'obj_recall': 0,
                'class_recall': 0,
                'total_queries': 0
            }
            results['multiple_views'][topk] = {
                'obj_recall': 0,
                'class_recall': 0,
                'total_queries': 0
            }
        
        # 遍历数据集进行查询
        dataloader = DataLoader(
            dataset, 
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        print("开始评估对象检索性能...")
        for i, batch in enumerate(tqdm(dataloader, desc="评估进度")):
            obj_id = batch['obj_id'][0]
            images = batch['images'][0]  # [num_views, num_images_per_view, C, H, W]
            
            # 单视图检索评估
            for view_idx in range(len(images)):
                view_images = images[view_idx]  # [num_images_per_view, C, H, W]
                
                # 使用视图中的第一张图像进行检索
                first_image = view_images[0]
                
                retrieval_results = self.query_by_single_view(
                    first_image,
                    topk=max(topk_list),
                    return_class_stats=True
                )
                
                if isinstance(retrieval_results, tuple):
                    results_list, stats = retrieval_results
                    
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
            
            # 多视图检索评估
            if use_multiple_views and len(images) >= 3:
                # 使用前三个视图进行多视图检索
                view1 = images[0][0]  # 每个视图取第一张图像
                view2 = images[1][0]
                view3 = images[2][0]
                
                retrieval_results = self.query_by_multiple_views(
                    view1, view2, view3,
                    topk=max(topk_list),
                    return_class_stats=True
                )
                
                if isinstance(retrieval_results, tuple):
                    results_list, stats = retrieval_results
                    
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
        
        # 计算平均召回率并记录到TensorBoard
        for mode in ['single_view', 'multiple_views']:
            for topk in topk_list:
                if results[mode][topk]['total_queries'] > 0:
                    results[mode][topk]['obj_recall'] /= results[mode][topk]['total_queries']
                    results[mode][topk]['class_recall'] /= results[mode][topk]['total_queries']
                    
                    # 记录到TensorBoard
                    self.writer.add_scalar(f'Evaluation/{mode}/obj_recall_top{topk}', results[mode][topk]['obj_recall'], eval_step)
                    self.writer.add_scalar(f'Evaluation/{mode}/class_recall_top{topk}', results[mode][topk]['class_recall'], eval_step)
        
        eval_step += 1
        
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
        
        # 初始化eval_step变量
        eval_step = 0
        
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
            batch_size=1,
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
                    exclude_view_index=view_idx,
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
        
        # 计算平均召回率并记录到TensorBoard
        for topk in topk_list:
            if results[topk]['total_queries'] > 0:
                results[topk]['view_recall'] /= results[topk]['total_queries']
            # 记录到TensorBoard（如果writer已初始化）
            if self.writer is not None:
                self.writer.add_scalar(f'Evaluation/view_retrieval/view_recall_top{topk}', results[topk]['view_recall'], eval_step)
        
        eval_step += 1
        
        return results