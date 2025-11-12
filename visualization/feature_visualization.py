#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import seaborn as sns

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 不再需要导入数据加载器，直接从特征数据库获取信息

class FeatureVisualizer:
    """
    特征可视化工具，用于将高维特征空间降维并可视化
    """
    def __init__(self, feature_db_path):
        """
        初始化可视化工具
        
        Args:
            feature_db_path: 特征数据库路径
        """
        self.feature_db_path = feature_db_path
        self.features = {}
        self.labels = {}
        self.colors = {}
        self.reduced_features = None
        
    def load_features(self, feat_type='obj_feat', max_objects_per_class=None, exclude_classes=None):
        """
        从数据库加载特征和标签
        
        Args:
            feat_type: 要可视化的特征类型 (obj_feat, global_features, view_features)
            max_objects_per_class: 每个类别最多加载的对象数量
            exclude_classes: 要排除的类别列表
        """
        print(f"从 {self.feature_db_path} 加载特征...")
        
        # 直接从HDF5数据库加载特征和创建类别映射，不依赖MultiViewDataset
        # 创建对象ID到类别的映射
        obj_to_class = {}
        
        # 从HDF5数据库加载特征
        with h5py.File(self.feature_db_path, 'r') as db:
            # 获取所有对象ID
            all_obj_ids = list(db.keys())
            
            # 为每个对象分配一个基于对象ID的类别
            # 可以根据obj_id的格式提取真实类别，这里使用对象ID的哈希值取模来分配类别
            for obj_id in all_obj_ids:
                # 尝试从对象ID中提取类别信息
                # 常见格式可能是 "class_name_0001" 或 "category_id_obj_id"
                try:
                    # 尝试几种常见的提取方式
                    if '_' in obj_id:
                        # 尝试按下划线分割，取第一个部分作为类别
                        cls = obj_id.split('_')[0]
                    else:
                        # 如果没有下划线，使用哈希值取模生成类别
                        cls = f"class_{hash(obj_id) % 10}"
                    obj_to_class[obj_id] = cls
                except Exception as e:
                    print(f"为对象 {obj_id} 分配类别时出错: {str(e)}")
                    # 出错时使用默认类别
                    cls = f"class_{hash(obj_id) % 10}"
                    obj_to_class[obj_id] = cls
        
        # 统计每个类别的对象数量
        class_counts = {}
        for obj_id, cls in obj_to_class.items():
            if cls not in class_counts:
                class_counts[cls] = 0
            class_counts[cls] += 1
        
        print(f"共找到 {len(obj_to_class)} 个对象，分布在 {len(class_counts)} 个类别中")
        print("类别统计:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count} 个对象")
        
        # 从HDF5数据库加载特征
        with h5py.File(self.feature_db_path, 'r') as db:
            # 计算每个类别要选择的对象数量
            selected_count = {}
            for obj_id, cls in obj_to_class.items():
                if exclude_classes and cls in exclude_classes:
                    continue
                    
                # 检查该对象是否在数据库中
                if obj_id not in db:
                    continue
                    
                # 检查是否已达到该类别最大对象数
                if max_objects_per_class:
                    if cls not in selected_count:
                        selected_count[cls] = 0
                    if selected_count[cls] >= max_objects_per_class:
                        continue
                    selected_count[cls] += 1
                
                try:
                    # 根据特征类型获取特征
                    if feat_type == 'obj_feat':
                        feat = db[obj_id][feat_type][()]
                    elif feat_type == 'global_features':
                        feat = db[obj_id][feat_type][()].squeeze()
                    elif feat_type == 'view_features':
                        # 对于view_features，取平均或第一个视图
                        view_feats = db[obj_id][feat_type][()]
                        feat = np.mean(view_feats, axis=0)
                    else:
                        raise ValueError(f"不支持的特征类型: {feat_type}")
                    
                    # 验证特征
                    if np.isnan(feat).any() or np.isinf(feat).any():
                        print(f"警告: 对象 {obj_id} 的特征包含 NaN 或 Inf 值，跳过")
                        continue
                    
                    self.features[obj_id] = feat
                    self.labels[obj_id] = cls
                    
                except Exception as e:
                    print(f"加载对象 {obj_id} 的特征时出错: {str(e)}")
        
        print(f"成功加载 {len(self.features)} 个对象的特征")
        
        # 为每个类别分配颜色
        unique_labels = list(set(self.labels.values()))
        color_map = cm.get_cmap('tab20', len(unique_labels))
        for i, label in enumerate(unique_labels):
            self.colors[label] = color_map(i)
        
        return self
    
    def reduce_dimension(self, method='tsne', n_components=2, perplexity=30, random_state=42):
        """
        使用降维方法将高维特征映射到低维空间
        
        Args:
            method: 降维方法 (tsne, pca, umap)
            n_components: 降维后的维度
            perplexity: t-SNE的困惑度参数
            random_state: 随机种子
        """
        if not self.features:
            raise ValueError("请先调用 load_features 方法加载特征")
        
        print(f"使用 {method} 进行降维，降维到 {n_components} 维...")
        
        # 准备特征矩阵
        obj_ids = list(self.features.keys())
        X = np.array([self.features[obj_id] for obj_id in obj_ids])
        
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            self.reduced_features = reducer.fit_transform(X)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
            self.reduced_features = reducer.fit_transform(X)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            self.reduced_features = reducer.fit_transform(X)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        print("降维完成")
        return self
    
    def plot_2d(self, output_path=None, title='特征空间可视化', figsize=(12, 10), 
                point_size=50, show_legend=True, legend_fontsize=10, alpha=0.7):
        """
        绘制2D可视化结果
        
        Args:
            output_path: 输出图像路径，如果为None则显示图像
            title: 图像标题
            figsize: 图像尺寸
            point_size: 点的大小
            show_legend: 是否显示图例
            legend_fontsize: 图例字体大小
            alpha: 点的透明度
        """
        if self.reduced_features is None or self.reduced_features.shape[1] != 2:
            raise ValueError("请先调用 reduce_dimension 方法，并且 n_components=2")
        
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=16)
        
        # 按类别绘制点
        obj_ids = list(self.features.keys())
        for i, obj_id in enumerate(obj_ids):
            label = self.labels[obj_id]
            color = self.colors[label]
            plt.scatter(
                self.reduced_features[i, 0], 
                self.reduced_features[i, 1],
                color=color,
                s=point_size,
                alpha=alpha,
                edgecolor='k',
                linewidth=0.5
            )
        
        # 添加图例
        if show_legend:
            unique_labels = list(set(self.labels.values()))
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors[lab], 
                               markersize=10, markeredgecolor='k') for lab in unique_labels]
            plt.legend(handles, unique_labels, fontsize=legend_fontsize, 
                     loc='best', bbox_to_anchor=(1.05, 1), title="类别")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存至: {output_path}")
        else:
            plt.show()
        
        return self
    
    def plot_3d(self, output_path=None, title='特征空间3D可视化', figsize=(12, 10), 
                point_size=50, show_legend=True, legend_fontsize=10, alpha=0.7):
        """
        绘制3D可视化结果
        
        Args:
            output_path: 输出图像路径，如果为None则显示图像
            title: 图像标题
            figsize: 图像尺寸
            point_size: 点的大小
            show_legend: 是否显示图例
            legend_fontsize: 图例字体大小
            alpha: 点的透明度
        """
        if self.reduced_features is None or self.reduced_features.shape[1] != 3:
            raise ValueError("请先调用 reduce_dimension 方法，并且 n_components=3")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontsize=16)
        
        # 按类别绘制点
        obj_ids = list(self.features.keys())
        for i, obj_id in enumerate(obj_ids):
            label = self.labels[obj_id]
            color = self.colors[label]
            ax.scatter(
                self.reduced_features[i, 0], 
                self.reduced_features[i, 1],
                self.reduced_features[i, 2],
                color=color,
                s=point_size,
                alpha=alpha,
                edgecolor='k',
                linewidth=0.5
            )
        
        # 添加图例
        if show_legend:
            unique_labels = list(set(self.labels.values()))
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors[lab], 
                               markersize=10, markeredgecolor='k') for lab in unique_labels]
            ax.legend(handles, unique_labels, fontsize=legend_fontsize, 
                     loc='best', title="类别")
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"3D可视化结果已保存至: {output_path}")
        else:
            plt.show()
        
        return self
    
    def generate_all_visualizations(self, output_dir='../visualizations', max_objects_per_class=50):
        """
        生成所有类型的可视化结果
        
        Args:
            output_dir: 输出目录
            max_objects_per_class: 每个类别最多可视化的对象数量
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载特征
        self.load_features(feat_type='obj_feat', max_objects_per_class=max_objects_per_class)
        
        # 使用不同降维方法生成可视化
        methods = ['tsne', 'pca', 'umap']
        for method in methods:
            print(f"\n使用 {method} 方法生成可视化...")
            # 2D可视化
            self.reduce_dimension(method=method, n_components=2)
            output_path = os.path.join(output_dir, f'feature_vis_{method}_2d.png')
            self.plot_2d(output_path=output_path, title=f'2D {method.upper()} 特征空间可视化')
            
            # 3D可视化
            self.reduce_dimension(method=method, n_components=3)
            output_path = os.path.join(output_dir, f'feature_vis_{method}_3d.png')
            self.plot_3d(output_path=output_path, title=f'3D {method.upper()} 特征空间可视化')
        
        print(f"\n所有可视化结果已保存至: {output_dir}")
        return self

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='特征空间可视化工具')
    
    # 特征数据库参数
    parser.add_argument('--feature_db', type=str, default='../features/feature_db.h5',
                        help='特征数据库路径')
    
    # 可视化参数
    parser.add_argument('--output_dir', type=str, default='../visualizations',
                        help='可视化结果保存目录')
    parser.add_argument('--method', type=str, default='all', choices=['tsne', 'pca', 'umap', 'all'],
                        help='降维方法')
    parser.add_argument('--n_components', type=int, default=2, choices=[2, 3],
                        help='降维后的维度')
    parser.add_argument('--max_objects_per_class', type=int, default=50,
                        help='每个类别最多可视化的对象数量')
    parser.add_argument('--feat_type', type=str, default='obj_feat', choices=['obj_feat', 'global_features', 'view_features'],
                        help='要可视化的特征类型')
    parser.add_argument('--show', action='store_true',
                        help='是否显示图像而不是保存')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析参数
    args = parse_args()
    
    # 创建可视化器
    visualizer = FeatureVisualizer(args.feature_db)
    
    if args.method == 'all':
        # 生成所有可视化
        visualizer.generate_all_visualizations(args.output_dir, args.max_objects_per_class)
    else:
        # 生成指定类型的可视化
        visualizer.load_features(
            feat_type=args.feat_type, 
            max_objects_per_class=args.max_objects_per_class
        )
        visualizer.reduce_dimension(
            method=args.method, 
            n_components=args.n_components
        )
        
        if args.output_dir and not args.show:
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(
                args.output_dir, 
                f'feature_vis_{args.feat_type}_{args.method}_{args.n_components}d.png'
            )
        else:
            output_path = None
        
        if args.n_components == 2:
            visualizer.plot_2d(
                output_path=output_path,
                title=f'2D {args.method.upper()} {args.feat_type} 特征空间可视化'
            )
        else:
            visualizer.plot_3d(
                output_path=output_path,
                title=f'3D {args.method.upper()} {args.feat_type} 特征空间可视化'
            )

if __name__ == "__main__":
    main()