#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的测试脚本，用于验证特征可视化模块是否能正常工作
如果特征数据库不存在，可以使用此脚本生成一个示例数据库
"""

import os
import sys
import h5py
import numpy as np
import argparse
from feature_visualization import FeatureVisualizer

def generate_sample_feature_db(output_path, num_classes=10, objects_per_class=5, feature_dim=256):
    """
    生成示例特征数据库用于测试
    
    Args:
        output_path: 输出文件路径
        num_classes: 类别数量
        objects_per_class: 每类对象数量
        feature_dim: 特征维度
    """
    print(f"生成示例特征数据库: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建HDF5文件
    with h5py.File(output_path, 'w') as db:
        for cls_idx in range(num_classes):
            cls_name = f"class_{cls_idx}"
            
            # 为每个类别生成中心特征
            center = np.random.randn(feature_dim)
            
            for obj_idx in range(objects_per_class):
                obj_id = f"{cls_name}_obj_{obj_idx}"
                
                # 在类别中心周围添加噪声生成对象特征
                obj_feat = center + 0.1 * np.random.randn(feature_dim)
                global_features = center + 0.05 * np.random.randn(feature_dim)
                view_features = np.array([center + 0.03 * np.random.randn(feature_dim) for _ in range(3)])
                
                # 创建对象组
                obj_group = db.create_group(obj_id)
                obj_group.create_dataset('obj_feat', data=obj_feat)
                obj_group.create_dataset('global_features', data=global_features)
                obj_group.create_dataset('view_features', data=view_features)
    
    print(f"示例特征数据库生成完成，包含 {num_classes} 个类别，每个类别 {objects_per_class} 个对象")
    return output_path

def test_visualization(feature_db_path, output_dir):
    """
    测试可视化功能
    """
    # 创建可视化器
    visualizer = FeatureVisualizer(feature_db_path)
    
    # 生成所有可视化
    visualizer.generate_all_visualizations(output_dir=output_dir, max_objects_per_class=10)
    
    print(f"测试完成! 可视化结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='测试特征可视化模块')
    parser.add_argument('--feature_db', type=str, default='../features/feature_db.h5',
                        help='特征数据库路径')
    parser.add_argument('--output_dir', type=str, default='../visualizations',
                        help='输出目录')
    parser.add_argument('--generate_sample', action='store_true',
                        help='是否生成示例特征数据库')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='示例数据库的类别数量')
    parser.add_argument('--objects_per_class', type=int, default=5,
                        help='示例数据库每类对象数量')
    
    args = parser.parse_args()
    
    # 如果需要生成示例数据库
    if args.generate_sample or not os.path.exists(args.feature_db):
        args.feature_db = generate_sample_feature_db(
            args.feature_db,
            args.num_classes,
            args.objects_per_class
        )
    else:
        print(f"使用现有特征数据库: {args.feature_db}")
    
    # 测试可视化
    test_visualization(args.feature_db, args.output_dir)

if __name__ == "__main__":
    main()