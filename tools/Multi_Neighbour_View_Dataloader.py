#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_views=3, num_images_per_view=5, split="train"):
        self.root_dir = root_dir
        self.transform = transform
        self.num_views = num_views   
        # 每个object下的viewpoints数量
        self.num_images_per_view = num_images_per_view
        # 每个viewpoint对应的邻域图数量
        self.split = split
        
        # 自动扫描类别列表
        self.obj_classes = self._get_all_classes()
        
        # 收集所有对象元信息（不加载图像）
        self.objects = []
        self.obj_to_idx = {}
        self.idx_to_obj = {}
        
        # 新增：缺失图片统计变量
        self.missing_stats = {
            'total_images': 0,        # 总图片数（理论值）
            'missing_images': 0,      # 缺失图片数
            'missing_per_class': {},  # 按类别统计的缺失数
            'missing_per_obj': {}     # 按对象统计的缺失数
        }
        
        # 初始化对象并统计缺失图片
        self._init_objects_and_count_missing()
        
        # 由于每个对象都恰好包含三个视点，不需要采样权重优化

    def _get_all_classes(self):
        """自动扫描数据集路径下的所有类别文件夹"""
        viewpoint_path = os.path.join(self.root_dir)
        if not os.path.exists(viewpoint_path):
            raise ValueError("路径不存在: {}".format(viewpoint_path))
        
        classes = []
        for item in os.listdir(viewpoint_path):
            item_path = os.path.join(viewpoint_path, item)
            split_path = os.path.join(item_path, self.split)
            if os.path.isdir(item_path) and os.path.exists(split_path):
                classes.append(item)
        
        if not classes:
            raise ValueError("在 {} 下未找到任何类别文件夹".format(viewpoint_path))
        return sorted(classes)

    def _init_objects_and_count_missing(self):
        """初始化对象元信息，并统计缺失图片"""
        obj_idx = 0
        # 初始化类别缺失统计
        for cls in self.obj_classes:
            self.missing_stats['missing_per_class'][cls] = 0
        
        for cls in self.obj_classes:
            cls_path = os.path.join(self.root_dir, '{}/{}'.format(cls, self.split))
            
            for obj_id in os.listdir(cls_path):
                obj_path = os.path.join(cls_path, obj_id)
                if not os.path.isdir(obj_path):
                    continue  # 跳过非目录文件
                
                # 获取可用视点
                available_views = self.get_available_views(obj_path)
                if len(available_views) < self.num_views:
                    continue  # 跳过视点不足的对象
                
                # 统计该对象的缺失图片数
                obj_missing = 0
                for view_num in available_views:
                    # 检查该视点下的所有图片，支持两位数字格式
                    if view_num < 10:
                        view_path = os.path.join(obj_path, 'view{:02d}'.format(view_num))
                    else:
                        view_path = os.path.join(obj_path, 'view{}'.format(view_num))
                    for img_idx in range(self.num_images_per_view):
                        img_name = "{}_{}.png".format(os.path.basename(obj_path), img_idx)
                        img_path = os.path.join(view_path, img_name)
                        
                        # 累计总图片数
                        self.missing_stats['total_images'] += 1
                        
                        # 检查是否缺失
                        if not os.path.exists(img_path):
                            # 打印缺失文件的路径
                            print(f"警告: 缺失文件 {img_path}")
                            obj_missing += 1
                            self.missing_stats['missing_images'] += 1
                            self.missing_stats['missing_per_class'][cls] += 1
                
                # 记录该对象的缺失数
                self.missing_stats['missing_per_obj'][obj_id] = obj_missing
                
                # 记录对象元信息
                # 修复：使用类别+obj_id作为唯一标识符，避免不同类别相同编号的对象冲突
                unique_obj_id = "{}_{}".format(cls, obj_id)
                obj_info = {
                    'class': cls,
                    'id': unique_obj_id,  # 使用唯一标识符
                    'original_id': obj_id,  # 保留原始ID
                    'path': obj_path,
                    'views': available_views,
                    'missing_images': obj_missing  # 新增：对象的缺失图片数
                }
                self.objects.append(obj_info)
                self.obj_to_idx[unique_obj_id] = obj_idx
                self.idx_to_obj[obj_idx] = unique_obj_id
                obj_idx += 1
        
        # 过滤掉只有一个有效视角的对象，确保InfoNCE损失有足够的正样本
        valid_objects = []
        removed_objects = 0
        for obj in self.objects:
            if len(obj['views']) >= 2:  # 至少需要2个视角才能形成正样本对
                valid_objects.append(obj)
            else:
                removed_objects += 1
                
        if removed_objects > 0:
            self.objects = valid_objects
            # 重建索引映射
            self.obj_to_idx = {obj['id']: idx for idx, obj in enumerate(self.objects)}
            self.idx_to_obj = {idx: obj['id'] for idx, obj in enumerate(self.objects)}
            print("过滤掉 {} 个视角不足的对象，剩余 {} 个对象".format(removed_objects, len(self.objects)))

        self.num_classes = len(self.objects)
        # 打印统计结果
        self._print_missing_stats()

    def _print_missing_stats(self):
        """打印缺失图片统计信息"""
        print("\n===== 缺失图片统计 =====")
        print("总图片数: {}".format(self.missing_stats['total_images']))
        print("缺失图片数: {}".format(self.missing_stats['missing_images']))
        
        # 添加除以零检查
        if self.missing_stats['total_images'] > 0:
            print("缺失比例: {:.2%}\n".format(self.missing_stats['missing_images'] / self.missing_stats['total_images']))
        else:
            print("缺失比例: 0.00% (总图片数为0)\n")
        
        print("按类别统计:")
        for cls, count in self.missing_stats['missing_per_class'].items():
            if count > 0:  # 只打印有缺失的类别
                print("  {}: {} 张缺失".format(cls, count))
        
        # 可选：打印缺失最多的前5个对象
        sorted_obj = sorted(
            self.missing_stats['missing_per_obj'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print("\n缺失最多的前5个对象:")
        for obj_id, count in sorted_obj:
            if count > 0:
                print("  {}: {} 张缺失".format(obj_id, count))
        print("=======================\n")

    def __len__(self):
        return len(self.objects)
    
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

    def load_view_images(self, obj_path, view_num):
        """加载指定视点的图片，确保返回张量"""
        # 支持两位数字格式，使用view01, view02等格式构建路径
        if view_num < 10:
            view_path = os.path.join(obj_path, 'view{:02d}'.format(view_num))
        else:
            view_path = os.path.join(obj_path, 'view{}'.format(view_num))
        images = []
        
        # 确保transform包含ToTensor()
        if self.transform is None:
            default_transform = transforms.Compose([transforms.ToTensor()])
        else:
            has_tensor = any(isinstance(t, transforms.ToTensor) for t in self.transform.transforms)
            if not has_tensor:
                new_transforms = list(self.transform.transforms) + [transforms.ToTensor()]
                default_transform = transforms.Compose(new_transforms)
            else:
                default_transform = self.transform
        
        for img_idx in range(self.num_images_per_view):
            img_name = "{}_{}.png".format(os.path.basename(obj_path), img_idx)
            img_path = os.path.join(view_path, img_name)
            
            try:
                # 不强制转换为RGB，保持原始模式（灰度图就是L模式）
                image = Image.open(img_path)
                # 检查是否为灰度图
                if image.mode == 'L':
                    # 灰度图：保持原始模式，后续在transform后扩展为三通道
                    pass
                else:
                    # 其他模式转换为RGB
                    image = image.convert('RGB')
                # 应用transform
                image = default_transform(image)
                # 检查是否为单通道（灰度图），如果是则扩展为三通道
                if image.size(0) == 1:  # C=1
                    # 将单通道扩展为三通道（复制三次）以匹配模型输入
                    image = image.repeat(3, 1, 1)
                images.append(image)
            except FileNotFoundError:
                # 打印缺失文件的路径
                print(f"警告: 缺失文件 {img_path}")
                if images:
                    images.append(images[-1].clone())
                else:
                    images.append(torch.zeros(3, 224, 224))
        
        return torch.stack(images)

        # 由于每个对象都恰好包含三个视点，删除了采样权重相关的方法
    
    def __getitem__(self, idx):
        """动态加载图像（懒加载）"""
        obj = self.objects[idx]
        obj_path = obj['path']
        
        # 随机选择num_views个视点
        selected_views = random.sample(obj['views'], self.num_views)
        
        # 动态加载图像
        view_images = [self.load_view_images(obj_path, view_num) for view_num in selected_views]
        all_views = torch.stack(view_images)  # (num_views, num_images_per_view, C, H, W)
        
        # 确保返回的obj_id是可以用于InfoNCE损失计算的格式
        # 这里我们返回对象ID作为字符串，在损失函数中会被正确处理
        return {
            'images': all_views,
            'obj_id': obj['id'],
            'view_numbers': selected_views,
            'obj_class': obj['class'],
            'label': self.obj_to_idx[obj['id']],
            'missing_images': obj['missing_images']  # 可选：返回该对象的缺失数
        }
