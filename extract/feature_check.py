#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import defaultdict

class FeatureValidator:
    """
    特征验证器，用于检查提取的特征是否正常
    """
    def __init__(self):
        # 初始化统计信息
        self.stats = {
            'total_samples': 0,
            'zero_vectors': defaultdict(int),  # 各特征类型的全零向量数量
            'identical_vectors': defaultdict(int),  # 各特征类型的全相等向量数量
            'extreme_values': defaultdict(int),  # 各特征类型的极端值异常数量
            'nan_values': defaultdict(int),  # 各特征类型的NaN值数量
            'inf_values': defaultdict(int),  # 各特征类型的Inf值数量
            'norm_stats': defaultdict(list),  # 各特征类型的范数统计
            'mean_stats': defaultdict(list),  # 各特征类型的均值统计
            'std_stats': defaultdict(list)    # 各特征类型的标准差统计
        }
        self.batch_stats = []  # 记录每个批次的统计信息
    
    def validate_features(self, features, obj_id):
        """
        验证单个对象的特征
        
        Args:
            features: 特征字典
            obj_id: 对象ID
        
        Returns:
            dict: 验证结果
        """
        results = {
            'obj_id': obj_id,
            'warnings': [],
            'stats': {}
        }
        
        self.stats['total_samples'] += 1
        
        # 验证每种特征类型
        for feat_name, feat_value in features.items():
            # 跳过非张量类型的特征
            if not isinstance(feat_value, torch.Tensor):
                continue
            
            # 展平多维特征以便分析
            flattened_feat = feat_value.reshape(-1)
            
            # 检查全零向量
            is_zero = torch.allclose(flattened_feat, torch.zeros_like(flattened_feat), atol=1e-6)
            if is_zero:
                self.stats['zero_vectors'][feat_name] += 1
                results['warnings'].append(f"{feat_name} 全为零向量")
            
            # 检查所有元素是否相等
            is_all_same = torch.allclose(flattened_feat, flattened_feat[0] * torch.ones_like(flattened_feat), atol=1e-6)
            if is_all_same and not is_zero:  # 排除全零的情况
                self.stats['identical_vectors'][feat_name] += 1
                results['warnings'].append(f"{feat_name} 所有元素值相等")
            
            # 检查极端值
            has_extreme = torch.any(torch.abs(flattened_feat) > 1e5)  # 绝对值大于1e5视为极端值
            if has_extreme:
                self.stats['extreme_values'][feat_name] += 1
                results['warnings'].append(f"{feat_name} 包含极端值")
            
            # 检查NaN值
            has_nan = torch.any(torch.isnan(flattened_feat))
            if has_nan:
                self.stats['nan_values'][feat_name] += 1
                results['warnings'].append(f"{feat_name} 包含NaN值")
            
            # 检查Inf值
            has_inf = torch.any(torch.isinf(flattened_feat))
            if has_inf:
                self.stats['inf_values'][feat_name] += 1
                results['warnings'].append(f"{feat_name} 包含Inf值")
            
            # 计算并记录统计信息
            with torch.no_grad():
                norm_val = torch.norm(flattened_feat).item()
                mean_val = torch.mean(flattened_feat).item()
                std_val = torch.std(flattened_feat).item()
                
                self.stats['norm_stats'][feat_name].append(norm_val)
                self.stats['mean_stats'][feat_name].append(mean_val)
                self.stats['std_stats'][feat_name].append(std_val)
            
            # 记录当前特征的统计信息
            results['stats'][feat_name] = {
                'norm': norm_val,
                'mean': mean_val,
                'std': std_val,
                'is_zero': is_zero,
                'is_all_same': is_all_same,
                'has_extreme': has_extreme,
                'has_nan': has_nan,
                'has_inf': has_inf
            }
        
        return results
    
    def validate_batch(self, batch_features, obj_ids):
        """
        验证一个批次的特征
        
        Args:
            batch_features: 批次特征字典列表
            obj_ids: 对象ID列表
        
        Returns:
            dict: 批次验证结果
        """
        batch_results = {
            'total_samples': len(obj_ids),
            'samples_with_warnings': 0,
            'warnings_by_type': defaultdict(int),
            'sample_results': []
        }
        
        # 验证每个样本
        for i, (features, obj_id) in enumerate(zip(batch_features, obj_ids)):
            sample_result = self.validate_features(features, obj_id)
            batch_results['sample_results'].append(sample_result)
            
            # 统计警告
            if sample_result['warnings']:
                batch_results['samples_with_warnings'] += 1
                for warning in sample_result['warnings']:
                    # 提取警告类型（前半部分）
                    warning_type = warning.split(' ')[0]
                    batch_results['warnings_by_type'][warning_type] += 1
        
        self.batch_stats.append(batch_results)
        return batch_results
    
    def get_batch_summary(self, batch_results):
        """
        获取批次验证摘要
        
        Args:
            batch_results: 批次验证结果
        
        Returns:
            str: 格式化的摘要字符串
        """
        summary = []
        summary.append(f"样本数: {batch_results['total_samples']}")
        summary.append(f"异常样本数: {batch_results['samples_with_warnings']}")
        
        if batch_results['warnings_by_type']:
            warning_strs = []
            for feat_type, count in batch_results['warnings_by_type'].items():
                warning_strs.append(f"{feat_type}: {count}")
            summary.append(f"异常类型: {', '.join(warning_strs)}")
        
        return " | ".join(summary)
    
    def get_overall_summary(self):
        """
        获取整体验证摘要
        
        Returns:
            dict: 整体验证统计信息
        """
        summary = {
            'total_samples': self.stats['total_samples'],
            'batch_count': len(self.batch_stats),
            'anomaly_summary': {},
            'feature_stats': {}
        }
        
        # 计算异常统计
        for anomaly_type in ['zero_vectors', 'identical_vectors', 'extreme_values', 'nan_values', 'inf_values']:
            anomaly_data = self.stats[anomaly_type]
            if anomaly_data:
                summary['anomaly_summary'][anomaly_type] = {}
                for feat_name, count in anomaly_data.items():
                    summary['anomaly_summary'][anomaly_type][feat_name] = {
                        'count': count,
                        'percentage': (count / self.stats['total_samples']) * 100 if self.stats['total_samples'] > 0 else 0
                    }
        
        # 计算特征统计
        for stat_type in ['norm_stats', 'mean_stats', 'std_stats']:
            stat_data = self.stats[stat_type]
            if stat_data:
                for feat_name, values in stat_data.items():
                    if feat_name not in summary['feature_stats']:
                        summary['feature_stats'][feat_name] = {}
                    
                    summary['feature_stats'][feat_name][stat_type.replace('_stats', '')] = {
                        'min': np.min(values) if values else 0,
                        'max': np.max(values) if values else 0,
                        'mean': np.mean(values) if values else 0,
                        'median': np.median(values) if values else 0
                    }
        
        return summary
    
    def print_overall_summary(self):
        """
        打印整体验证摘要
        """
        summary = self.get_overall_summary()
        
        print("\n" + "="*80)
        print("特征验证整体摘要")
        print("="*80)
        print(f"总样本数: {summary['total_samples']}")
        print(f"总批次: {summary['batch_count']}")
        
        # 打印异常统计
        if summary['anomaly_summary']:
            print("\n异常统计:")
            print("-"*40)
            for anomaly_type, feat_data in summary['anomaly_summary'].items():
                print(f"{anomaly_type.replace('_', ' ').title()}:")
                for feat_name, stats in feat_data.items():
                    print(f"  {feat_name}: {stats['count']}个 ({stats['percentage']:.2f}%)")
        
        # 打印特征统计
        if summary['feature_stats']:
            print("\n特征统计:")
            print("-"*40)
            for feat_name, stats in summary['feature_stats'].items():
                print(f"{feat_name}:")
                for stat_type, values in stats.items():
                    print(f"  {stat_type}: 最小值={values['min']:.4f}, 最大值={values['max']:.4f}, "
                          f"均值={values['mean']:.4f}, 中位数={values['median']:.4f}")
        
        # 总体评估
        print("\n总体评估:")
        print("-"*40)
        has_nan = any(feat_data for feat_data in summary['anomaly_summary'].get('nan_values', {}).values())
        has_inf = any(feat_data for feat_data in summary['anomaly_summary'].get('inf_values', {}).values())
        has_extreme = any(feat_data for feat_data in summary['anomaly_summary'].get('extreme_values', {}).values())
        high_zero_rate = any(stats['percentage'] > 50 for feat_type_data in summary['anomaly_summary'].get('zero_vectors', {}).values() for stats in [feat_type_data])
        
        if has_nan or has_inf:
            print("⚠️  严重问题: 特征中包含NaN或Inf值，需要检查模型和数据!")
        elif has_extreme:
            print("⚠️  警告: 特征中包含极端值，建议检查模型训练情况!")
        elif high_zero_rate:
            print("⚠️  警告: 超过50%的特征为零向量，可能存在模型问题!")
        elif summary['anomaly_summary']:
            print("ℹ️  注意: 存在少量异常特征，但总体情况可接受。")
        else:
            print("✅  特征质量良好，未发现明显异常。")
        
        print("="*80)

def create_feature_dict(view_features, mid_features, global_features, obj_feat):
    """
    创建统一格式的特征字典
    
    Args:
        view_features: 视点特征
        mid_features: 中间特征
        global_features: 全局特征
        obj_feat: 对象特征
    
    Returns:
        dict: 格式化的特征字典
    """
    return {
        'view_features': view_features,
        'mid_features': mid_features,
        'global_features': global_features,
        'obj_feat': obj_feat
    }