#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np

class PreTrainMultiViewDataset(Dataset):
    """
    预训练多视图数据集，用于类别分类任务
    从文件结构中读取多视图图像并提取类别标签
    """
    def __init__(self, root_dir, transform=None, num_views=3, num_images_per_view=5, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views
        self.num_images_per_view = num_images_per_view
        self.split = split
        
        # 存储数据样本信息
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.use_alternative_structure = False  # 默认使用标准结构
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载数据样本和类别信息，支持标准结构和替代结构"""
        # 获取所有类别
        self.obj_classes = self._get_all_classes()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.obj_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # 根据结构类型选择不同的加载逻辑
        if self.use_alternative_structure:
            # 替代结构：root_dir/class_name/viewXX/
            self._load_data_alternative_structure()
        else:
            # 标准结构：root_dir/class_name/split/obj_id/viewXX/
            self._load_data_standard_structure()
        
        print(f"加载了 {len(self.samples)} 个样本，共 {len(self.obj_classes)} 个类别")
    
    def _load_data_standard_structure(self):
        """加载标准结构的数据：root_dir/class_name/split/obj_id/viewXX/"""
        # 遍历每个类别
        for class_name in self.obj_classes:
            class_idx = self.class_to_idx[class_name]
            # 构建类别下的split目录路径
            split_dir = os.path.join(self.root_dir, class_name, self.split)
            
            if not os.path.exists(split_dir):
                print(f"警告: 类别 {class_name} 的 {self.split} 文件夹不存在，跳过")
                continue
            
            # 遍历每个对象
            for obj_name in sorted(os.listdir(split_dir)):
                obj_dir = os.path.join(split_dir, obj_name)
                if not os.path.isdir(obj_dir):
                    continue
                
                # 获取可用的视图
                available_views = self.get_available_views(obj_dir)
                if len(available_views) < self.num_views:
                    continue  # 跳过视图不足的对象
                
                # 收集所有视图的图像
                view_images = []
                for view_num in available_views[:self.num_views]:  # 只取前num_views个视图
                    # 支持两位数字格式
                    if view_num < 10:
                        view_dir = os.path.join(obj_dir, f'view{view_num:02d}')
                    else:
                        view_dir = os.path.join(obj_dir, f'view{view_num}')
                    
                    if not os.path.exists(view_dir):
                        continue
                    
                    # 获取该视图下的所有图像（按照固定命名格式）
                    images = []
                    for img_idx in range(self.num_images_per_view):
                        img_name = f"{obj_name}_{img_idx}.png"
                        img_path = os.path.join(view_dir, img_name)
                        if os.path.exists(img_path):
                            images.append(img_path)
                        else:
                            # 如果文件不存在，记录缺失
                            print(f"警告: 缺失文件 {img_path}")
                    
                    # 如果图像数量不足，重复最后一个图像
                    while len(images) < self.num_images_per_view and images:
                        images.append(images[-1])
                    
                    # 如果还是没有图像，跳过此视图
                    if not images:
                        continue
                    
                    view_images.append(images)
                
                # 确保有足够的视图
                if len(view_images) >= self.num_views:
                    self.samples.append({
                        'obj_id': f"{class_name}_{obj_name}",
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'view_images': view_images[:self.num_views]  # 确保只取num_views个视图
                    })
    
    def _load_data_alternative_structure(self):
        """加载替代结构的数据：root_dir/class_name/viewXX/"""
        # 遍历每个类别
        for class_name in self.obj_classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root_dir, class_name)
            
            # 在替代结构中，类别目录直接包含view文件夹
            available_views = self.get_available_views(class_dir)
            if len(available_views) < self.num_views:
                print(f"警告: 类别 {class_name} 的视图数量不足，跳过")
                continue
            
            # 在替代结构中，每个类别作为一个对象处理
            obj_name = class_name  # 使用类别名作为对象名
            
            # 收集所有视图的图像
            view_images = []
            for view_num in available_views[:self.num_views]:  # 只取前num_views个视图
                # 支持两位数字格式
                if view_num < 10:
                    view_dir = os.path.join(class_dir, f'view{view_num:02d}')
                else:
                    view_dir = os.path.join(class_dir, f'view{view_num}')
                
                if not os.path.exists(view_dir):
                    continue
                
                # 在替代结构中，尝试获取所有图像文件
                images = []
                # 尝试不同的命名模式
                patterns = [
                    f"{obj_name}_{{}}.png",       # 类别名_索引.png
                    f"{class_name}_{{}}.png",     # 同样的类别名_索引.png
                    "{}.png",                     # 索引.png
                ]
                
                # 尝试最多20张图像
                for img_idx in range(20):
                    found = False
                    for pattern in patterns:
                        try:
                            img_name = pattern.format(img_idx)
                            img_path = os.path.join(view_dir, img_name)
                            if os.path.exists(img_path):
                                images.append(img_path)
                                found = True
                                break
                        except:
                            continue
                    
                    # 如果找到足够的图像，停止搜索
                    if len(images) >= self.num_images_per_view:
                        break
                
                # 如果没有找到匹配的图像，尝试获取目录中的所有图像
                if not images:
                    try:
                        all_images = [os.path.join(view_dir, f) for f in os.listdir(view_dir) 
                                    if f.endswith(('.png', '.jpg', '.jpeg'))]
                        if all_images:
                            images = sorted(all_images)[:self.num_images_per_view]
                    except:
                        pass
                
                # 如果图像数量不足，重复最后一个图像
                while len(images) < self.num_images_per_view and images:
                    images.append(images[-1])
                
                # 如果还是没有图像，跳过此视图
                if not images:
                    continue
                
                view_images.append(images)
            
            # 确保有足够的视图
            if len(view_images) >= self.num_views:
                self.samples.append({
                    'obj_id': f"{class_name}_{obj_name}",
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'view_images': view_images[:self.num_views]  # 确保只取num_views个视图
                })
    
    def get_available_views(self, obj_path):
        """获取对象的所有可用视点"""
        views = []
        for item in os.listdir(obj_path):
            if item.startswith('view') and os.path.isdir(os.path.join(obj_path, item)):
                try:
                    # 支持view01-view03等两位数字格式
                    view_num = int(item.replace('view', ''))
                    views.append(view_num)
                except ValueError:
                    continue
        return sorted(views)
    
    def _get_all_classes(self):
        """增强版类别检测，自动适应多种数据集结构格式"""
        viewpoint_path = self.root_dir
        if not os.path.exists(viewpoint_path):
            raise ValueError(f"路径不存在: {viewpoint_path}")
        
        classes = []
        found_structure = "未确定"
        
        # 策略1: 检查是否存在 root_dir/split/class_name 结构 (ModelNet常见格式)
        split_dir = os.path.join(viewpoint_path, self.split)
        if os.path.exists(split_dir) and os.path.isdir(split_dir):
            potential_classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
            if potential_classes:
                classes = potential_classes
                found_structure = f"root_dir/{self.split}/class_name"
                self.use_alternative_structure = False
                print(f"检测到结构: {found_structure}")
                print(f"找到 {len(classes)} 个类别用于 {self.split} 集")
                return sorted(classes)
        
        # 策略2: 尝试标准结构 root_dir/class_name/split
        for item in os.listdir(viewpoint_path):
            item_path = os.path.join(viewpoint_path, item)
            split_path = os.path.join(item_path, self.split)
            if os.path.isdir(item_path) and os.path.exists(split_path):
                classes.append(item)
        
        if classes:
            found_structure = "root_dir/class_name/split"
            self.use_alternative_structure = False
            print(f"检测到结构: {found_structure}")
        else:
            # 策略3: 对于验证集，尝试其他可能的验证集名称
            if self.split == 'val':
                for alt_split in ['test', 'validation']:
                    alt_classes = []
                    for item in os.listdir(viewpoint_path):
                        item_path = os.path.join(viewpoint_path, item)
                        split_path = os.path.join(item_path, alt_split)
                        if os.path.isdir(item_path) and os.path.exists(split_path):
                            alt_classes.append(item)
                    
                    if alt_classes:
                        classes = alt_classes
                        self.split = alt_split
                        found_structure = f"root_dir/class_name/{alt_split}"
                        self.use_alternative_structure = False
                        print(f"警告: 未找到'val'文件夹，使用'{alt_split}'文件夹作为验证集")
                        print(f"检测到结构: {found_structure}")
                        break
            
            # 策略4: 尝试替代结构 root_dir/class_name/viewXX/ (直接包含视图)
            if not classes:
                print(f"尝试替代路径结构...")
                for item in os.listdir(viewpoint_path):
                    item_path = os.path.join(viewpoint_path, item)
                    if os.path.isdir(item_path):
                        # 检查是否包含view开头的文件夹
                        try:
                            has_views = any(f.startswith('view') for f in os.listdir(item_path) 
                                          if os.path.isdir(os.path.join(item_path, f)))
                            if has_views:
                                classes.append(item)
                        except:
                            continue
                
                if classes:
                    found_structure = "root_dir/class_name/viewXX/"
                    self.use_alternative_structure = True
                    print(f"检测到替代结构: {found_structure}")
                else:
                    # 策略5: 尝试直接将所有文件夹作为类别
                    all_dirs = [d for d in os.listdir(viewpoint_path) 
                               if os.path.isdir(os.path.join(viewpoint_path, d))]
                    if all_dirs:
                        classes = all_dirs
                        found_structure = "root_dir/class_name/ (假定为类别)"
                        self.use_alternative_structure = True  # 假设这是替代结构
                        print(f"未检测到标准结构，使用所有文件夹作为类别: {len(classes)}个")
                        print(f"假定结构: {found_structure}")
                    else:
                        # 最后尝试: 检查是否存在data子文件夹
                        data_dir = os.path.join(viewpoint_path, 'data')
                        if os.path.exists(data_dir) and os.path.isdir(data_dir):
                            data_classes = [d for d in os.listdir(data_dir) 
                                           if os.path.isdir(os.path.join(data_dir, d))]
                            if data_classes:
                                classes = data_classes
                                found_structure = "root_dir/data/class_name/"
                                self.use_alternative_structure = False
                                print(f"检测到data子文件夹结构: {found_structure}")
                            else:
                                raise ValueError(f"在 {viewpoint_path} 下未找到任何类别文件夹或data子文件夹")
                        else:
                            raise ValueError(f"在 {viewpoint_path} 下未找到任何类别文件夹。数据集结构可能不符合预期。")
        
        print(f"找到 {len(classes)} 个类别用于 {self.split} 集，结构类型: {found_structure}")
        return sorted(classes)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        obj_id = sample['obj_id']
        class_idx = sample['class_idx']
        view_images = sample['view_images']
        
        # 加载所有视图的图像
        images = []
        for view_files in view_images:
            view_imgs = []
            for img_path in view_files:
                try:
                    # 加载图像
                    img = Image.open(img_path)
                    # 检查是否为灰度图
                    if img.mode == 'L':
                        # 灰度图：保持原始模式
                        pass
                    else:
                        # 其他模式转换为RGB
                        img = img.convert('RGB')
                    # 应用变换
                    if self.transform:
                        img = self.transform(img)
                    # 检查是否为单通道（灰度图），如果是则扩展为三通道
                    if img.size(0) == 1:  # C=1
                        img = img.repeat(3, 1, 1)
                    view_imgs.append(img)
                except Exception as e:
                    print(f"加载图像 {img_path} 时出错: {e}")
                    # 创建一个零张量作为替代
                    view_imgs.append(torch.zeros(3, 224, 224))
            
            # 确保有足够的图像
            while len(view_imgs) < self.num_images_per_view:
                view_imgs.append(view_imgs[-1].clone() if view_imgs else torch.zeros(3, 224, 224))
            
            # 转换为张量并添加到视图列表
            view_imgs_tensor = torch.stack(view_imgs)
            images.append(view_imgs_tensor)
        
        # 堆叠所有视图
        images_tensor = torch.stack(images)
        
        return {
            'obj_id': obj_id,
            'images': images_tensor,  # [num_views, num_images_per_view, C, H, W]
            'class_idx': class_idx,
            'class_name': sample['class_name']
        }

def get_pre_train_dataloaders(root_dir, batch_size=32, image_size=224, num_workers=4, num_views=3, num_images_per_view=5):
    """
    获取预训练的数据加载器，增强版支持多种数据集结构和鲁棒的验证集创建
    
    Args:
        root_dir: 数据集根目录
        batch_size: 批次大小
        image_size: 图像尺寸
        num_workers: 数据加载进程数
        num_views: 每个对象使用的视图数量
        num_images_per_view: 每个视图使用的图像数量
    
    Returns:
        包含训练/验证数据加载器和类别映射的字典
    """
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集变换
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"开始加载数据集，根目录: {root_dir}")
    
    # 创建训练数据集
    train_dataset = PreTrainMultiViewDataset(
        root_dir=root_dir,
        transform=train_transform,
        num_views=num_views,
        num_images_per_view=num_images_per_view,
        split='train'
    )
    
    # 记录训练数据集是否使用替代结构
    use_alternative_structure = train_dataset.use_alternative_structure
    
    # 创建验证集的增强策略
    val_dataset = None
    
    # 策略1: 对于替代结构，直接从训练数据中分割验证集
    if use_alternative_structure:
        print("检测到替代数据结构，将从训练数据中分割验证集")
        import random
        random.seed(42)  # 固定随机种子以确保可重复性
        
        # 计算验证集大小
        total_samples = len(train_dataset)
        val_size = max(100, int(total_samples * 0.1))  # 最小100个样本，或者10%
        
        # 创建索引列表
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        # 分割索引
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # 使用Subset创建数据集
        from torch.utils.data import Subset
        
        # 创建训练子集
        train_subset = Subset(train_dataset, train_indices)
        
        # 创建验证子集，并应用验证变换
        # 为了应用不同的变换，我们需要一个自定义的Subset类
        class TransformableSubset(Subset):
            def __init__(self, dataset, indices, transform):
                super().__init__(dataset, indices)
                self.transform = transform
                self.orig_transform = dataset.transform
            
            def __getitem__(self, idx):
                # 临时替换变换
                self.dataset.transform = self.transform
                try:
                    sample = super().__getitem__(idx)
                finally:
                    # 恢复原始变换
                    self.dataset.transform = self.orig_transform
                return sample
        
        val_dataset = TransformableSubset(train_dataset, val_indices, val_transform)
        train_dataset = TransformableSubset(train_dataset, train_indices, train_transform)
        
        print(f"从替代结构数据中分割: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")
    else:
        # 策略2: 尝试创建独立的验证集
        val_dataset = None
        val_splits = ['val', 'test', 'validation']
        
        for split_name in val_splits:
            try:
                print(f"尝试创建基于'{split_name}'的验证集...")
                val_dataset = PreTrainMultiViewDataset(
                    root_dir=root_dir,
                    transform=val_transform,
                    num_views=num_views,
                    num_images_per_view=num_images_per_view,
                    split=split_name
                )
                
                # 检查验证集是否有足够的样本
                if len(val_dataset) > 0:
                    print(f"成功创建基于'{split_name}'的验证集，包含 {len(val_dataset)} 个样本")
                    break
                else:
                    print(f"'{split_name}'验证集存在但为空，尝试下一个...")
                    val_dataset = None
            except Exception as e:
                print(f"创建{split_name}验证集失败: {e}")
        
        # 策略3: 如果所有独立验证集都失败，从训练集中分割
        if val_dataset is None or len(val_dataset) == 0:
            print("警告: 未找到合适的验证集，将从训练集中随机分割10%作为验证集")
            import random
            random.seed(42)
            
            # 计算验证集大小
            total_samples = len(train_dataset)
            val_size = max(100, int(total_samples * 0.1))
            
            # 创建索引列表并打乱
            indices = list(range(total_samples))
            random.shuffle(indices)
            
            # 分割索引
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            
            # 创建训练子集
            class TransformableSubset(Subset):
                def __init__(self, dataset, indices, transform):
                    super().__init__(dataset, indices)
                    self.transform = transform
                    self.orig_transform = dataset.transform
                
                def __getitem__(self, idx):
                    # 临时替换变换
                    self.dataset.transform = self.transform
                    try:
                        sample = super().__getitem__(idx)
                    finally:
                        # 恢复原始变换
                        self.dataset.transform = self.orig_transform
                    return sample
            
            # 使用自定义的Subset类以支持不同的变换
            train_subset = TransformableSubset(train_dataset, train_indices, train_transform)
            val_dataset = TransformableSubset(train_dataset, val_indices, val_transform)
            
            print(f"从训练集中分割的验证集，大小: {len(val_dataset)} 个样本")
    
    # 确保我们有有效的训练集和验证集
    if not hasattr(train_dataset, '__len__') or len(train_dataset) == 0:
        raise RuntimeError("训练集为空或创建失败")
    if not hasattr(val_dataset, '__len__') or len(val_dataset) == 0:
        raise RuntimeError("验证集为空或创建失败")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 获取类别映射（从原始训练数据集）
    class_to_idx = getattr(train_dataset, 'class_to_idx', {})
    idx_to_class = getattr(train_dataset, 'idx_to_class', {})
    
    # 如果是Subset，尝试从原始数据集获取类别映射
    if isinstance(train_dataset, torch.utils.data.Subset) and hasattr(train_dataset.dataset, 'class_to_idx'):
        class_to_idx = train_dataset.dataset.class_to_idx
        idx_to_class = train_dataset.dataset.idx_to_class
    
    print(f"数据加载器创建完成: 训练集批次 {len(train_loader)}, 验证集批次 {len(val_loader)}")
    print(f"类别数量: {len(class_to_idx)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class
    }

def get_class_statistics(dataset):
    """
    获取数据集中各类别的统计信息
    """
    class_counts = {}
    for sample in dataset.samples:
        class_name = sample['class_name']
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    return class_counts