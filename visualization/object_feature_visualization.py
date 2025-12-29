#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征数据库可视化脚本 - Object级别特征分布
将特征数据库中的每一个对象特征表述为空间点，用红色小球呈现
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

# 检测并设置中文字体支持
def setup_chinese_font():
    """
    设置matplotlib的中文字体支持
    """
    import matplotlib.font_manager as fm
    
    # 常见的中文字体列表
    chinese_fonts = [
        'SimHei',  # Windows 黑体
        'Microsoft YaHei',  # 微软雅黑
        'SimSun',  # Windows 宋体
        'KaiTi',   # 楷体
        'FangSong', # 仿宋
        'DejaVu Sans',  # 备用字体
        'Arial Unicode MS',  # Mac OS 字体
    ]
    
    # 检查系统可用字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            print(f"使用字体: {font}")
            return True
    
    print("警告: 未找到合适的中文字体，图表中的中文可能无法正常显示")
    return False

setup_chinese_font()

def load_object_features(feature_db_path, feat_type='obj_feat'):
    """
    从特征数据库加载对象特征
    
    Args:
        feature_db_path: 特征数据库路径
        feat_type: 特征类型 ('obj_feat', 'global_features', 'view_features')
    
    Returns:
        特征矩阵 (numpy array)、对象ID列表和类别列表
    """
    features = []
    obj_ids = []
    class_labels = []
    
    with h5py.File(feature_db_path, 'r') as db:
        for obj_id in db.keys():
            try:
                if feat_type in db[obj_id]:
                    if feat_type == 'obj_feat':
                        feat = db[obj_id][feat_type][()]
                    elif feat_type == 'global_features':
                        feat = db[obj_id][feat_type][()].squeeze()
                    elif feat_type == 'view_features':
                        # 对于view_features，取平均
                        view_feats = db[obj_id][feat_type][()]
                        feat = np.mean(view_feats, axis=0)
                    else:
                        raise ValueError(f"不支持的特征类型: {feat_type}")
                    
                    # 验证特征
                    if not (np.isnan(feat).any() or np.isinf(feat).any()):
                        features.append(feat)
                        obj_ids.append(obj_id)
                        
                        # 从对象ID中提取类别信息
                        # 假设对象ID格式为 "class_name_0001" 或类似格式
                        if '_' in obj_id:
                            class_name = obj_id.split('_')[0]
                        else:
                            # 如果没有下划线，使用对象ID的哈希值取模生成类别
                            class_name = f"class_{hash(obj_id) % 40}"
                        class_labels.append(class_name)
                else:
                    print(f"警告: 对象 {obj_id} 中没有 {feat_type} 特征")
            except Exception as e:
                print(f"加载对象 {obj_id} 的特征时出错: {str(e)}")
    
    if not features:
        raise ValueError(f"没有加载到任何 {feat_type} 特征")
    
    features = np.array(features)
    print(f"成功加载 {len(features)} 个对象的 {feat_type} 特征")
    print(f"特征矩阵形状: {features.shape}")
    print(f"检测到 {len(set(class_labels))} 个不同类别")
    
    return features, obj_ids, class_labels

def visualize_3d_scatter(features, class_labels, title="Object级别特征空间分布", save_path=None):
    """
    3D散点图可视化特征分布，使用不同颜色表示不同类别
    
    Args:
        features: 特征矩阵 (N, D)
        class_labels: 类别标签列表
        title: 图表标题
        save_path: 保存路径
    """
    # 如果特征维度大于3，使用PCA降维到3D
    if features.shape[1] > 3:
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features)
        print(f"使用PCA将{features.shape[1]}维特征降维到3D")
    elif features.shape[1] == 2:
        # 2D特征扩展到3D（第三维设为0）
        features_3d = np.hstack([features, np.zeros((features.shape[0], 1))])
    else:
        features_3d = features

    # 获取唯一类别并生成颜色映射
    unique_classes = list(set(class_labels))
    n_classes = len(unique_classes)
    
    # 生成颜色映射
    colors = plt.colormaps.get_cmap('tab20') if hasattr(plt.colormaps, 'get_cmap') else plt.cm.get_cmap('tab20')
    class_to_color = {cls: colors(i % colors.N) for i, cls in enumerate(unique_classes)}
    
    # 创建3D图形
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 为每个类别分别绘制点
    for i, class_name in enumerate(unique_classes):
        # 获取当前类别的索引
        class_indices = [idx for idx, label in enumerate(class_labels) if label == class_name]
        
        # 提取当前类别的特征
        class_features = features_3d[class_indices]
        
        # 绘制当前类别的点
        ax.scatter(
            class_features[:, 0], 
            class_features[:, 1], 
            class_features[:, 2],
            c=[class_to_color[class_name]], 
            s=20,  # 小球大小
            alpha=0.7,  # 透明度
            edgecolors='darkred',  # 边框颜色
            linewidth=0.5,  # 边框宽度
            label=class_name
        )
    
    ax.set_xlabel('特征维度 1')
    ax.set_ylabel('特征维度 2')
    ax.set_zlabel('特征维度 3')
    ax.set_title(title)
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例（如果类别数量不是太多）
    if n_classes <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D可视化图已保存至: {save_path}")
    
    plt.show()

def visualize_2d_scatter(features, class_labels, title="Object级别特征空间分布", method='pca', save_path=None):
    """
    2D散点图可视化特征分布，使用不同颜色表示不同类别
    
    Args:
        features: 特征矩阵 (N, D)
        class_labels: 类别标签列表
        title: 图表标题
        method: 降维方法 ('pca' 或 'tsne')
        save_path: 保存路径
    """
    # 如果特征维度大于2，使用PCA或t-SNE降维到2D
    if features.shape[1] > 2:
        if method == 'pca':
            reducer = PCA(n_components=2)
            features_2d = reducer.fit_transform(features)
            print(f"使用PCA将{features.shape[1]}维特征降维到2D")
        elif method == 'tsne':
            # 对于大数据集，可能需要调整参数
            n_samples = min(features.shape[0], 1000)  # 限制样本数量以提高效率
            sample_indices = np.random.choice(features.shape[0], n_samples, replace=False)
            sampled_features = features[sample_indices]
            sampled_labels = [class_labels[i] for i in sample_indices]  # 同时采样标签
            
            n_components = min(2, sampled_features.shape[0] - 1)  # 确保样本数大于n_components
            if n_components >= 2:
                perplexity = min(30, sampled_features.shape[0] - 1)
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                features_2d = reducer.fit_transform(sampled_features)
                print(f"使用t-SNE将{features.shape[1]}维特征降维到2D (使用{sampled_features.shape[0]}个样本)")
                class_labels = sampled_labels  # 使用采样后的标签
            else:
                # 如果样本数不足，使用PCA作为备选
                reducer = PCA(n_components=2)
                features_2d = reducer.fit_transform(features)
                print(f"样本数不足，使用PCA将{features.shape[1]}维特征降维到2D")
        else:
            raise ValueError(f"不支持的降维方法: {method}")
    else:
        features_2d = features

    # 获取唯一类别并生成颜色映射
    unique_classes = list(set(class_labels))
    n_classes = len(unique_classes)
    
    # 生成颜色映射
    colors = plt.colormaps.get_cmap('tab20') if hasattr(plt.colormaps, 'get_cmap') else plt.cm.get_cmap('tab20')
    class_to_color = {cls: colors(i % colors.N) for i, cls in enumerate(unique_classes)}
    
    # 创建2D图形
    plt.figure(figsize=(12, 8))
    
    # 为每个类别分别绘制点
    for i, class_name in enumerate(unique_classes):
        # 获取当前类别的索引
        class_indices = [idx for idx, label in enumerate(class_labels) if label == class_name]
        
        # 提取当前类别的特征
        class_features = features_2d[class_indices]
        
        # 绘制当前类别的点
        plt.scatter(
            class_features[:, 0], 
            class_features[:, 1],
            c=[class_to_color[class_name]], 
            s=15,  # 点大小
            alpha=0.7,  # 透明度
            edgecolors='darkred',  # 边框颜色
            linewidth=0.3,  # 边框宽度
            label=class_name
        )
    
    plt.xlabel('特征维度 1')
    plt.ylabel('特征维度 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 添加图例（如果类别数量不是太多）
    if n_classes <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{method.upper()}可视化图已保存至: {save_path}")
    
    plt.show()

def analyze_feature_distribution(features):
    """
    分析特征分布统计信息
    
    Args:
        features: 特征矩阵 (N, D)
    """
    print("=== Object级别特征分布统计 ===")
    print(f"特征矩阵形状: {features.shape}")
    print(f"特征维度: {features.shape[1]}")
    print(f"对象数量: {features.shape[0]}")
    
    print(f"特征均值: {np.mean(features):.4f}")
    print(f"特征标准差: {np.std(features):.4f}")
    print(f"特征最小值: {np.min(features):.4f}")
    print(f"特征最大值: {np.max(features):.4f}")
    
    # 计算特征之间的距离统计
    if features.shape[0] > 1:
        # 计算特征点之间的距离（仅对前1000个点进行采样以提高效率）
        sample_size = min(1000, features.shape[0])
        sample_features = features[:sample_size]
        
        # 计算距离矩阵
        dist_matrix = np.sqrt(((sample_features[:, None, :] - sample_features[None, :, :]) ** 2).sum(axis=2))
        # 排除对角线（自己与自己的距离）
        dist_matrix = dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)].reshape(dist_matrix.shape[0], -1)
        
        print(f"对象间平均距离: {np.mean(dist_matrix):.4f}")
        print(f"对象间距离标准差: {np.std(dist_matrix):.4f}")
        print(f"对象间最小距离: {np.min(dist_matrix):.4f}")
        print(f"对象间最大距离: {np.max(dist_matrix):.4f}")
        print(f"对象间距离分布 - 25%: {np.percentile(dist_matrix, 25):.4f}, 50%: {np.percentile(dist_matrix, 50):.4f}, 75%: {np.percentile(dist_matrix, 75):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Object级别特征数据库可视化脚本")
    parser.add_argument("--feature_db_path", type=str, required=True,
                        help="特征数据库路径 (.h5)")
    parser.add_argument("--feat_type", type=str, default="obj_feat",
                        choices=['obj_feat', 'global_features', 'view_features'],
                        help="特征类型")
    parser.add_argument("--output_dir", type=str, default="./visualizations",
                        help="输出目录路径")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大采样数量（如果指定，将从特征中随机采样）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载特征
    print(f"正在从数据库加载特征: {args.feature_db_path}")
    features, obj_ids, class_labels = load_object_features(args.feature_db_path, args.feat_type)
    
    # 如果指定了采样数量，则随机采样
    if args.max_samples and args.max_samples < features.shape[0]:
        indices = np.random.choice(features.shape[0], args.max_samples, replace=False)
        features = features[indices]
        obj_ids = [obj_ids[i] for i in indices]
        class_labels = [class_labels[i] for i in indices]
        print(f"从{len(obj_ids)}个对象中随机采样了{len(features)}个")
    
    # 分析特征分布
    analyze_feature_distribution(features)
    
    # 生成可视化
    base_name = f"obj_features_{args.feat_type}"
    
    # 3D可视化
    visualize_3d_scatter(
        features, 
        class_labels,
        title=f"{base_name} - 3D Object级别特征空间分布", 
        save_path=os.path.join(args.output_dir, f"{base_name}_3d_scatter.png")
    )
    
    # 2D PCA可视化
    visualize_2d_scatter(
        features, 
        class_labels,
        title=f"{base_name} - 2D PCA Object级别特征空间分布", 
        method='pca',
        save_path=os.path.join(args.output_dir, f"{base_name}_pca_2d.png")
    )
    
    # 2D t-SNE可视化
    visualize_2d_scatter(
        features, 
        class_labels,
        title=f"{base_name} - 2D t-SNE Object级别特征空间分布", 
        method='tsne',
        save_path=os.path.join(args.output_dir, f"{base_name}_tsne_2d.png")
    )
    
    print("Object级别特征可视化完成！")

if __name__ == "__main__":
    main()