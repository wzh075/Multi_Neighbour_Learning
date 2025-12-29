#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式检索脚本
允许用户通过命令行选择类别和obj_id来确定数据集train子集中的一个object，
输出该object在三种检索模式下返回的前十个answer的class与obj_id，并且支持反复查询。
"""

# 修改导入语句，使用直接导入的方式
# 修改导入语句，使其在不同环境下都能正确工作
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 直接从evaluate模块导入所需的函数和类
from evaluate import (
    load_model,
    single_viewgroup_retrieval,
    multi_viewgroup_retrieval,
    viewgroup_cross_retrieval,
    FeatureDatabase
)

# 导入其他必要的模块
from tools.Multi_Neighbour_View_Dataloader import MultiViewDataset
import argparse
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm


def get_user_input(prompt, valid_options):
    """
    获取用户输入并验证
    
    Args:
        prompt: 提示信息
        valid_options: 有效选项列表
        
    Returns:
        用户选择的有效选项
    """
    while True:
        user_input = input(prompt).strip()
        if user_input in valid_options:
            return user_input
        else:
            print(f"无效输入，请输入以下选项之一: {', '.join(valid_options)}")


def get_user_input_int(prompt, valid_range=None):
    """
    获取用户整数输入并验证
    
    Args:
        prompt: 提示信息
        valid_range: 有效范围 (min, max)
        
    Returns:
        用户选择的有效整数
    """
    while True:
        user_input = input(prompt).strip()
        try:
            user_input = int(user_input)
            if valid_range is None or (valid_range[0] <= user_input <= valid_range[1]):
                return user_input
            else:
                print(f"无效输入，请输入 {valid_range[0]} 到 {valid_range[1]} 之间的整数")
        except ValueError:
            print("无效输入，请输入整数")


def display_classes(dataset):
    """
    显示可用的类别列表
    
    Args:
        dataset: MultiViewDataset实例
        
    Returns:
        类别列表
    """
    print("\n可用的类别:")
    for i, cls in enumerate(dataset.obj_classes, 1):
        print(f"{i}. {cls}")
    return dataset.obj_classes


def display_obj_ids(dataset, selected_class):
    """
    显示指定类别下的所有obj_id
    
    Args:
        dataset: MultiViewDataset实例
        selected_class: 选择的类别
        
    Returns:
        指定类别下的obj_id列表
    """
    obj_ids = []
    print(f"\n类别 '{selected_class}' 下的obj_id:")
    
    for i, obj in enumerate(dataset.objects, 1):
        if obj['class'] == selected_class:
            obj_id = obj['original_id']
            obj_ids.append(obj_id)
            print(f"{i}. {obj_id}")
    
    return obj_ids


def get_object_by_class_and_id(dataset, selected_class, selected_obj_id):
    """
    根据类别和obj_id获取对象
    
    Args:
        dataset: MultiViewDataset实例
        selected_class: 选择的类别
        selected_obj_id: 选择的obj_id
        
    Returns:
        对象信息和索引
    """
    for idx, obj in enumerate(dataset.objects):
        if obj['class'] == selected_class and obj['original_id'] == selected_obj_id:
            return obj, idx
    return None, None


def retrieve_object(model, feature_db, dataset, obj_idx, device, transform, num_views, top_k=10):
    """
    使用三种检索模式检索对象
    
    Args:
        model: 评估模型
        feature_db: 特征数据库
        dataset: 训练数据集
        obj_idx: 对象索引
        device: 计算设备
        transform: 图像变换
        num_views: 训练时设置的视点组数量
        top_k: 返回前k个结果
        
    Returns:
        三种检索模式的结果
    """
    # 获取对象数据
    data = dataset[obj_idx]
    obj_id = data['obj_id']
    images = data['images']  # [num_views, num_images_per_view, C, H, W]
    
    # 获取数据库特征和类别映射
    object_db = feature_db.get_object_features()
    view_db = feature_db.get_view_features()
    class_map = feature_db.get_class_map()
    
    # 转换数据库特征为张量
    def convert_db_to_tensor(db):
        db_ids = list(db.keys())
        db_feats = torch.stack([db[db_id] for db_id in db_ids])
        return db_ids, db_feats
    
    object_db_ids, object_db_feats = convert_db_to_tensor(object_db)
    view_db_ids, view_db_feats = convert_db_to_tensor(view_db)
    
    # 确保数据库特征在同一设备上
    object_db_feats = object_db_feats.to(device)
    view_db_feats = view_db_feats.to(device)
    
    results = {}
    
    # 1. 单视点组检索
    print("\n=== 单视点组检索结果 ===")
    # 只使用第一个视点组
    viewgroup_images = images[0]
    single_view_feat = single_viewgroup_retrieval(model, viewgroup_images, num_views, device, transform)
    
    # 计算相似度
    similarities = torch.nn.functional.cosine_similarity(
        single_view_feat.unsqueeze(0), object_db_feats, dim=1
    )
    
    # 获取top-k结果
    _, indices = torch.sort(similarities, descending=True)
    # 确保indices是一维张量
    if indices.dim() > 1:
        indices = indices.squeeze()
    single_view_results = []
    for i in range(min(top_k, len(indices))):  # 确保不超过indices的长度
        idx = indices[i].item()  # 现在应该可以安全地调用.item()
        result_id = object_db_ids[idx]
        result_class = class_map[result_id]
        # 提取obj_id（去掉类别前缀）
        result_obj_id = result_id.split('_')[1]
        single_view_results.append((result_class, result_obj_id))
        print(f"{i+1}. class: {result_class}, obj_id: {result_obj_id}")
    
    results['single_view'] = single_view_results
    
    # 2. 多视点组检索
    print("\n=== 多视点组检索结果 ===")
    # 使用所有视点组
    viewgroup_images_list = [images[j] for j in range(images.shape[0])]
    multi_view_feat = multi_viewgroup_retrieval(model, viewgroup_images_list, device, transform)
    
    # 计算相似度
    similarities = torch.nn.functional.cosine_similarity(
        multi_view_feat.unsqueeze(0), object_db_feats, dim=1
    )
    
    # 获取top-k结果
    _, indices = torch.sort(similarities, descending=True)
    # 确保indices是一维张量
    if indices.dim() > 1:
        indices = indices.squeeze()
    multi_view_results = []
    for i in range(min(top_k, len(indices))):  # 确保不超过indices的长度
        idx = indices[i].item()  # 现在应该可以安全地调用.item()
        result_id = object_db_ids[idx]
        result_class = class_map[result_id]
        # 提取obj_id（去掉类别前缀）
        result_obj_id = result_id.split('_')[1]
        multi_view_results.append((result_class, result_obj_id))
        print(f"{i+1}. class: {result_class}, obj_id: {result_obj_id}")
    
    results['multi_view'] = multi_view_results
    
    # 3. 视点组互检索
    print("\n=== 视点组互检索结果 ===")
    # 使用第一个视点组
    viewgroup_images = images[0]
    view_feat = viewgroup_cross_retrieval(model, viewgroup_images, device, transform)
    
    # 计算相似度
    similarities = torch.nn.functional.cosine_similarity(
        view_feat.unsqueeze(0), view_db_feats, dim=1
    )
    
    # 获取top-k结果
    _, indices = torch.sort(similarities, descending=True)
    # 确保indices是一维张量
    if indices.dim() > 1:
        indices = indices.squeeze()
    view_cross_results = []
    for i in range(min(top_k, len(indices))):  # 确保不超过indices的长度
        idx = indices[i].item()  # 现在应该可以安全地调用.item()
        result_id = view_db_ids[idx]
        result_class = class_map[result_id]
        # 提取obj_id（去掉类别前缀和_view_x后缀）
        result_obj_id = result_id.split('_')[1]
        view_cross_results.append((result_class, result_obj_id))
        print(f"{i+1}. class: {result_class}, obj_id: {result_obj_id}")
    
    results['view_cross'] = view_cross_results
    
    return results


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='交互式多视图检索系统')
    
    # 数据参数
    parser.add_argument('--root_dir', type=str, default='/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0',
                        help='数据集根目录')
    parser.add_argument('--split', type=str, default='train',
                        help='数据集分割 (train/val/test)')
    parser.add_argument('--num_views', type=int, default=3,
                        help='训练时设置的视点组数量')
    parser.add_argument('--num_images_per_view', type=int, default=5,
                        help='每个视点组的图像数量')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像尺寸')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='../checkpoints/best_model.pth',
                        help='训练得到的模型pth文件路径')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度大小')
    parser.add_argument('--num_classes', type=int, default=40,
                        help='类别数量')
    
    # 特征数据库参数
    parser.add_argument('--feature_db_path', type=str, default='../features/feature_db.h5',
                        help='特征数据库路径')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
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
    
    # 创建训练数据集
    dataset = MultiViewDataset(
        root_dir=args.root_dir,
        transform=transform,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view,
        split=args.split
    )
    
    print(f"训练数据集加载完成: {len(dataset)}个样本")
    
    # 交互式查询循环
    while True:
        # 显示可用类别
        classes = display_classes(dataset)
        
        # 让用户选择类别
        class_choice = get_user_input_int(
            "\n请选择类别编号: ",
            valid_range=(1, len(classes))
        )
        selected_class = classes[class_choice - 1]
        
        # 显示该类别下的obj_id
        obj_ids = display_obj_ids(dataset, selected_class)
        
        # 让用户选择obj_id
        obj_choice = get_user_input_int(
            "\n请选择obj_id编号: ",
            valid_range=(1, len(obj_ids))
        )
        selected_obj_id = obj_ids[obj_choice - 1]
        
        # 获取对象
        obj, obj_idx = get_object_by_class_and_id(dataset, selected_class, selected_obj_id)
        if obj is None:
            print("未找到该对象，请重新选择")
            continue
        
        print(f"\n已选择对象: 类别 = {selected_class}, obj_id = {selected_obj_id}")
        
        # 检索对象
        retrieve_object(model, feature_db, dataset, obj_idx, device, transform, args.num_views)
        
        # 询问用户是否继续查询
        continue_query = get_user_input(
            "\n是否继续查询? (y/n): ",
            valid_options=['y', 'n', 'Y', 'N']
        )
        
        if continue_query.lower() != 'y':
            break
    
    # 关闭特征数据库
    feature_db.close()
    print("\n=== 查询结束 ===")


if __name__ == "__main__":
    main()











