#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# 添加当前目录到Python路径，确保能正确导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import numpy as np
import json
import h5py
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torch import nn

from tools.obj_views_dataloader import MultiViewDataset
from models.myCorrNet import MultiViewRetrievalModel
from Cross_loss import MultiViewSplitLoss
from retrieval_sys import RetrievalSystem
from mytrain import train_model


class FeatureDatabase:
    """特征数据库，用于存储和检索特征"""

    def __init__(self, db_path, mode='a'):
        """
        :param db_path: 数据库文件路径
        :param mode: 打开模式 ('r'只读, 'a'追加, 'w'写入)
        """
        self.db_path = db_path
        self.mode = mode
        self.db = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        """打开数据库"""
        self.db = h5py.File(self.db_path, self.mode)

    def close(self):
        """关闭数据库"""
        if self.db:
            self.db.close()
            self.db = None

    def store_features(self, obj_id, features):
        """
        存储特征到数据库
        :param obj_id: 物体ID
        :param features: 特征字典，包含:
            - 'view_features': 视点特征 [num_views, feat_dim]
            - 'mid_features': 中间特征 [num_views, feat_dim]
            - 'obj_feat': 物体特征 [feat_dim]
        """
        if not self.db:
            self.open()

        # 创建或更新对象组
        if obj_id in self.db:
            obj_group = self.db[obj_id]
        else:
            obj_group = self.db.create_group(obj_id)

        # 存储特征
        for key, value in features.items():
            if key in obj_group:
                del obj_group[key]  # 删除现有数据集
            obj_group.create_dataset(key, data=value.cpu().numpy())

    def retrieve_features(self, obj_id):
        """
        从数据库检索特征
        :param obj_id: 物体ID
        :return: 特征字典
        """
        if not self.db:
            self.open()

        if obj_id not in self.db:
            return None

        obj_group = self.db[obj_id]
        features = {
            'view_features': torch.tensor(obj_group['view_features'][:]),
            'mid_features': torch.tensor(obj_group['mid_features'][:]),
            'obj_feat': torch.tensor(obj_group['obj_feat'][:]),
        }
        # 如果存在global_features，也添加到返回字典中
        if 'global_features' in obj_group:
            features['global_features'] = torch.tensor(obj_group['global_features'][:])
        return features

    def get_all_features(self):
        """
        获取数据库中所有特征
        :return: 特征字典 {obj_id: features}
        """
        if not self.db:
            self.open()

        all_features = {}
        for obj_id in self.db.keys():
            all_features[obj_id] = self.retrieve_features(obj_id)
        return all_features


def train_model_func(args):
    """训练模型的功能函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备: {}".format(device))

    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print("检测到{}块可用GPU".format(num_gpus))

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset = MultiViewDataset(
        root_dir=args.root_dir,
        transform=transform,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view,
        split="train"
    )

    # 初始化模型
    model = MultiViewRetrievalModel(
        num_classes=dataset.num_classes,
        feat_dim=args.feat_dim
    )

    # 禁用调试模式以减少输出
    model.view_global_fusion.debug = False

    # 多GPU支持 - 正确的顺序：先确保模型在主设备上，再包装DataParallel
    if num_gpus > 1:
        print("使用 {} 块GPU进行并行训练".format(num_gpus))
        # DataParallel要求模型先在device_ids[0]上
        device_ids = list(range(num_gpus))
        model = model.to(device_ids[0])  # 确保模型在第一个GPU上
        model = nn.DataParallel(model, device_ids=device_ids)
        # 对于DataParallel包装的模型，需要通过.module访问原始模型属性
        model.module.view_global_fusion.debug = False
    else:
        # 单GPU情况
        model = model.to(device)

    # 初始化损失函数
    criterion = MultiViewSplitLoss(
        tau=args.tau,
        lambda_view_sim=args.lambda_view_sim,
        lambda_global_consistency=args.lambda_global_consistency,
        lambda_obj_cluster=args.lambda_obj_cluster,
        device=device
    )

    print("开始训练模型...")
    print("训练集样本数: {}".format(len(dataset)))
    print("类别数: {}".format(dataset.num_classes))
    print("训练参数 - 轮数: {}, 批大小: {}, 学习率: {}".format(args.epochs, args.batch_size, args.lr))

    # 训练模型
    print("=" * 50)
    print("开始训练模型...")
    print("=" * 50)

    trained_model = train_model(
        model,
        dataset,
        criterion,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device=device
    )

    return trained_model


def extract_features_func(args, model=None):
    """提取特征并保存到特征数据库"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备: {}".format(device))

    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print("检测到{}块可用GPU".format(num_gpus))

    # 确保特征数据库目录存在
    db_dir = os.path.dirname(args.feature_db)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    # 加载模型（如果没有提供）
    if model is None:
        model = load_retrieval_model(args, device)
    try:
        # 获取模型第一个可训练参数
        first_param = next(model.parameters())
        # 确保缩进正确（与周围代码块一致）
        print("模型首个参数均值: {}".format(first_param.mean().item()))
        print("模型首个参数标准差: {}".format(first_param.std().item()))
    except StopIteration:
        print("警告：模型没有可训练参数（可能未正确初始化）")

    # 多GPU支持 - 正确的顺序：先确保模型在主设备上，再包装DataParallel
    if num_gpus > 1:
        print("使用 {} 块GPU进行并行处理".format(num_gpus))
        # DataParallel要求模型先在device_ids[0]上
        device_ids = list(range(num_gpus))
        model = model.to(device_ids[0])  # 确保模型在第一个GPU上
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        # 单GPU情况
        model = model.to(device)

    # 设置模型为评估模式
    model.eval()
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    dataset = MultiViewDataset(
        root_dir=args.root_dir,
        transform=transform,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view,
        split="train"  # 提取所有数据
    )

    # 创建特征数据库
    os.makedirs(os.path.dirname(args.feature_db), exist_ok=True)

    # 提取并存储特征
    with FeatureDatabase(args.feature_db, mode='w') as db:
        print("=" * 50)
        print("开始提取特征并存储到数据库: {}".format(args.feature_db))
        print("=" * 50)

        # 使用DataLoader批量处理
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="特征提取")):
            # 获取数据
            obj_ids = batch['obj_id']
            images = batch['images']  # [batch_size, num_views, num_images_per_view, C, H, W]

            # 显示当前batch的信息
            print("\nBatch {0}: 包含{1}个样本".format(batch_idx, len(obj_ids)))
            print(" Batch样本ID: {0}".format(obj_ids))

            # 分离三个视点组并移动到设备
            view1 = images[:, 0].to(device)  # [batch_size, num_images_per_view, C, H, W]
            view2 = images[:, 1].to(device)
            view3 = images[:, 2].to(device)

            # 前向传播
            print("  正在处理Batch {0}的特征提取...".format(batch_idx))
            with torch.no_grad():
                outputs = model(view1, view2, view3)

            # 提取特征
            view_features = outputs['view_feats']
            mid_features = outputs['mid_feat']
            global_features = outputs['global_feat']
            obj_feats = outputs['obj_feat']

            # 存储特征并展示特征信息
            for i, obj_id in enumerate(obj_ids):
                # 获取特征并计算统计信息
                obj_feat = obj_feats[i]
                global_feat = global_features[i]

                obj_feat_mean = obj_feat.mean().item()
                obj_feat_std = obj_feat.std().item()
                obj_feat_norm = torch.norm(obj_feat).item()
                global_feat_mean = global_feat.mean().item()
                global_feat_std = global_feat.std().item()

                # 检查obj_feat是否全为零
                is_zero = torch.allclose(obj_feat, torch.zeros_like(obj_feat), atol=1e-6)

                print("   样本 {0}/{1} - {2}: ".format(i + 1, len(obj_ids), obj_id))
                print(
                    "     obj_feat 统计: 均值={0:.6f}, 标准差={1:.6f}, 范数={2:.6f}".format(obj_feat_mean, obj_feat_std,
                                                                                            obj_feat_norm))
                if is_zero:
                    print("     ⚠️ 警告: obj_feat全为零向量!")

                # 显示特征值（截断显示）
                feat_values = obj_feat.cpu().numpy()
                if len(feat_values) > 20:
                    first_ten = ["{0:.4f}".format(val) for val in feat_values[:10]]
                    last_ten = ["{0:.4f}".format(val) for val in feat_values[-10:]]
                    print("     obj_feat 值: [{0}], ..., [{1}]".format(', '.join(first_ten), ', '.join(last_ten)))
                else:
                    print("     obj_feat 值: [{0}]".format(', '.join(["{0:.4f}".format(val) for val in feat_values])))

                print("     global_feat 统计: 均值={0:.6f}, 标准差={1:.6f}".format(global_feat_mean, global_feat_std))

            # 存储特征
            for i, obj_id in enumerate(obj_ids):
                features = {
                    'view_features': view_features[i].cpu(),
                    'mid_features': mid_features[i].cpu().unsqueeze(0),
                    'global_features': global_features[i].cpu().unsqueeze(0),
                    'obj_feat': obj_feats[i].cpu(),
                }
                db.store_features(obj_id, features)

    print("特征提取完成，存储到: {}".format(args.feature_db))


def load_retrieval_model(args, device):
    """加载用于检索的模型"""
    print("加载检索模型...")

    # 初始化模型
    model = MultiViewRetrievalModel(
        num_classes=1,  # 占位值，检索时不重要
        feat_dim=args.feat_dim
    )

    model_path = args.model_path

    # 如果未提供模型路径，尝试使用默认路径
    if not model_path:
        default_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(default_model_path):
            model_path = default_model_path
            print("使用默认模型路径: {}".format(model_path))

    # 加载预训练权重
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # 处理DataParallel保存的权重
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        print("已加载模型权重: {}".format(model_path))
    else:
        print("警告: 未找到模型文件，使用随机初始化模型")

    # 注意：不在这里调用model.to(device)，而是让调用函数决定何时以及如何移动模型到设备
    return model


def retrieval_func(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print("检测到{}块可用GPU".format(num_gpus))

    # 加载模型
    model = load_retrieval_model(args, device)  # 新增函数加载模型

    # 多GPU支持 - 正确的顺序：先确保模型在主设备上，再包装DataParallel
    if num_gpus > 1:
        print("使用 {} 块GPU进行并行处理".format(num_gpus))
        # DataParallel要求模型先在device_ids[0]上
        device_ids = list(range(num_gpus))
        model = model.to(device_ids[0])  # 确保模型在第一个GPU上
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        # 单GPU情况
        model = model.to(device)

    # 创建数据集（用于元数据和示例检索）
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiViewDataset(
        root_dir=args.root_dir,
        transform=transform,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view,
        split="train"
    )

    # 初始化检索系统
    retrieval_sys = RetrievalSystem(
        feature_db=args.feature_db,
        dataset=dataset if not args.feature_db else None,
        device=device,
        model=model
    )

    # 创建保存目录
    os.makedirs("retrieval_results", exist_ok=True)

    # 示例检索
    if args.example_retrieval:
        sample_idx = 0  # 使用第一个样本作为查询
        sample = dataset[sample_idx]

        # 1. 使用单个视点组查询
        view_idx = 0  # 使用第一个视点组
        view1 = sample['images'][view_idx]
        results = retrieval_sys.query_by_single_view(view1, topk=args.retrieval_top_k if hasattr(args,
                                                                                                 'retrieval_top_k') else 5)

        print("\n使用单个视点组查询结果:")
        for obj_id, sim in results:
            print("Obj ID: {} - 相似度: {:.4f}".format(obj_id, sim))

        # 可视化单个视点查询结果（暂时跳过）
        # 由于visualize_retrieval函数未定义，且可视化部分可以先不考虑
        # save_path = "retrieval_results/single_view_query_{}_view{}.png".format(sample['obj_id'], view_idx)
        # query_images = [Image.fromarray((img * 255).byte().permute(1, 2, 0).cpu().numpy())
        #                for img in view1]
        # visualize_retrieval(
        #     query_images,
        #     [(obj_id, 0, sim) for obj_id, sim in results],
        #     dataset,
        #     save_path,
        #     title="Single View Query - Obj {} (View {})".format(sample['obj_id'], view_idx)
        # )

        # 2. 使用多个视点组查询
        views = [sample['images'][i] for i in range(args.num_views)]
        results = retrieval_sys.query_by_multiple_views(*views, topk=args.retrieval_top_k if hasattr(args,
                                                                                                     'retrieval_top_k') else 5)

        print("\n使用{}个视点组查询结果:".format(args.num_views))
        for obj_id, sim in results:
            print("Obj ID: {} - 相似度: {:.4f}".format(obj_id, sim))

        # 可视化多视点查询结果
        save_path = "retrieval_results/multi_view_query_{}.png".format(sample['obj_id'])

        # 创建查询图像网格（所有视点）
        query_images = []
        for i in range(args.num_views):
            query_images.extend([
                Image.fromarray((img * 255).byte().permute(1, 2, 0).cpu().numpy())
                for img in views[i]
            ])

        # 由于visualize_retrieval函数未定义，且可视化部分可以先不考虑
        # visualize_retrieval(
        #     query_images,
        #     [(obj_id, 0, sim) for obj_id, sim in results],
        #     dataset,
        #     save_path,
        #     title="Multi View Query - Obj {} ({} Views)".format(sample['obj_id'], args.num_views)
        # )

        # 3. 视点间检索
        view_idx = 0  # 使用第一个视点
        view1 = sample['images'][view_idx]
        similar_views = retrieval_sys.query_by_view_for_same_obj_views(view1, obj_id=sample['obj_id'],
                                                                       topk=args.retrieval_top_k if hasattr(args,
                                                                                                            'retrieval_top_k') else 5)

        print("\n相似视点检索结果:")
        for obj_id, view_idx, sim in similar_views:
            print("Obj ID: {} - 视点 {} - 相似度: {:.4f}".format(obj_id, view_idx, sim))

        # 可视化视点间检索结果
        save_path = "retrieval_results/view_retrieval_{}_view{}.png".format(sample['obj_id'], view_idx)

        # 获取查询视点图像
        query_images = [
            Image.fromarray((img * 255).byte().permute(1, 2, 0).cpu().numpy())
            for img in sample['images'][view_idx]
        ]

        # 由于visualize_retrieval函数未定义，且可视化部分可以先不考虑
        # visualize_retrieval(
        #     query_images,
        #     similar_views,
        #     dataset,
        #     save_path,
        #     title="View Retrieval - Obj {} (View {})".format(sample['obj_id'], view_idx)
        # )


def evaluate_func(args):
    """评估检索系统性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count()
    print("检测到{}块可用GPU".format(num_gpus))

    # 加载模型
    model = load_retrieval_model(args, device)

    # 多GPU支持
    if num_gpus > 1:
        print("使用 {} 块GPU进行并行处理".format(num_gpus))
        device_ids = list(range(num_gpus))
        model = model.to(device_ids[0])
        model = nn.DataParallel(model, device_ids=device_ids)
    else:
        model = model.to(device)

    # 创建数据集
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MultiViewDataset(
        root_dir=args.root_dir,
        transform=transform,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view,
        split="train"
    )

    # 初始化检索系统
    retrieval_sys = RetrievalSystem(
        feature_db=args.feature_db,
        dataset=dataset if not args.feature_db else None,
        device=device,
        model=model
    )

    # 创建保存目录
    os.makedirs("evaluation_results", exist_ok=True)

    print("=" * 50)
    print("开始评估检索系统性能")
    print("=" * 50)

    # 评估对象检索性能
    print("\n1. 评估对象检索性能...")
    obj_results = retrieval_sys.evaluate_object_retrieval(
        dataset,
        topk_list=args.eval_topk_list,
        use_single_view=args.eval_single_view,
        use_multiple_views=args.eval_multiple_views
    )

    # 打印对象检索结果
    print("\n对象检索结果:")
    print("-" * 40)
    for mode in ['single_view', 'multiple_views', 'overall']:
        if mode in obj_results and obj_results[mode]:
            print("\n{0}:".format(mode.replace('_', ' ').title()))
            for topk in args.eval_topk_list:
                if topk in obj_results[mode]:
                    stats = obj_results[mode][topk]
                    if 'total_queries' in stats and stats['total_queries'] > 0:
                        print("  Top-{0}:".format(topk))
                        if 'obj_recall' in stats:
                            print("    Object Recall: {0:.4f}".format(stats['obj_recall']))
                        if 'class_recall' in stats:
                            print("    Class Recall: {0:.4f}".format(stats['class_recall']))

    # 评估视图检索性能
    view_results = {}
    if args.eval_view_retrieval:
        print("\n2. 评估视图检索性能...")
        view_results = retrieval_sys.evaluate_view_retrieval(
            dataset,
            topk_list=args.eval_topk_list
        )

        # 打印视图检索结果
        print("\n视图检索结果:")
        print("-" * 40)
        for topk in args.eval_topk_list:
            if topk in view_results:
                stats = view_results[topk]
                if 'total_queries' in stats and stats['total_queries'] > 0:
                    print("Top-{0}:".format(topk))
                    if 'view_recall' in stats:
                        print("  View Recall: {0:.4f}".format(stats['view_recall']))

    # 保存评估结果
    results_file = os.path.join("evaluation_results", "evaluation_results.json")
    all_results = {
        'object_retrieval': obj_results,
        'view_retrieval': view_results
    }

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n评估结果已保存到: {}".format(results_file))
    print("=" * 50)
    print("评估完成")
    print("=" * 50)


def main(args):
    # 根据模式执行不同功能
    if args.mode == 'train':
        trained_model = train_model_func(args)

        # 训练后直接提取特征
        if args.extract_after_train:
            args.model_path = os.path.join(args.save_dir, 'best_model.pth')
            args.num_objects = len(MultiViewDataset(
                root_dir=args.root_dir,
                num_views=args.num_views,
                num_images_per_view=args.num_images_per_view,
                split="train"
            ))
            extract_features_func(args, trained_model)

    elif args.mode == 'extract':
        # 需要先获取数据集大小
        dataset = MultiViewDataset(
            root_dir=args.root_dir,
            num_views=args.num_views,
            num_images_per_view=args.num_images_per_view,
            split="train"
        )
        args.num_objects = len(dataset)
        extract_features_func(args)

    elif args.mode == 'retrieve':
        retrieval_func(args)

    elif args.mode == 'evaluate':
        evaluate_func(args)

    else:
        raise ValueError("未知模式: {}".format(args.mode))


if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 将父目录添加到系统路径
    sys.path.append(current_dir)

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='多视图检索系统')

    # 全局参数
    parser.add_argument('--mode', choices=['train', 'extract', 'retrieve', 'evaluate'], required=True,
                        help='运行模式: train(训练), extract(特征提取), retrieve(检索), evaluate(评估)')

    # 数据参数
    parser.add_argument('--root_dir', type=str, default='/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0',
                        help='数据集根目录')
    parser.add_argument('--obj_classes', nargs='+', default=['airplane', 'bench', 'bowl', 'cone', 'desk',
                                                             'flower', 'keyboard', 'mantel',
                                                             'person', 'radio', 'sofa', 'table', 'tv',
                                                             'xbox', 'bathtub', 'bookshelf', 'car',
                                                             'cup', 'door', 'glass', 'lamp', 'monitor',
                                                             'piano', 'range', 'stairs', 'tent',
                                                             'vase', 'bed', 'bottle', 'chair', 'curtain',
                                                             'dresser', 'guitar', 'laptop', 'night',
                                                             'plant', 'sink', 'stool', 'toilet', 'wardrobe'],
                        help='要使用的物体类别列表')
    parser.add_argument('--num_views', type=int, default=3,
                        help='每个物体的视点数量')
    parser.add_argument('--num_images_per_view', type=int, default=5,
                        help='每个视点的图像数量')
    parser.add_argument('--image_size', type=int, default=224,
                        help='输入图像尺寸')

    # 模型参数
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度大小')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='InfoNCE损失的温度参数')
    parser.add_argument('--lambda_intra', type=float, default=0.2,
                        help='视点内损失的权重系数')
    parser.add_argument('--lambda_center', type=float, default=1.0,
                        help='object中心损失的权重系数')
    parser.add_argument('--lambda_inter', type=float, default=1.5,
                        help='object间损失的权重系数')
    parser.add_argument('--inter_margin', type=float, default=1.0,
                        help='最小距离阈值')
    parser.add_argument('--lambda_view_sim', type=float, default=0.0001,
                        help='视图特征间相似度损失权重（设置极小）')
    parser.add_argument('--lambda_global_consistency', type=float, default=1.0,
                        help='global_feat一致性损失权重')
    parser.add_argument('--lambda_obj_cluster', type=float, default=1.0,
                        help='obj_feat聚类损失权重')
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    parser.add_argument('--extract_after_train', action='store_true',
                        help='训练后自动提取特征')
    parser.add_argument('--num_classes', type=int, default=0,
                        help='类别数量，0表示自动检测')

    # 特征提取参数
    parser.add_argument('--feature_db', type=str, default='features/feature_db.h5',
                        help='特征数据库路径')
    parser.add_argument('--num_objects', type=int, default=0,
                        help='数据集中的物体数量（自动计算）')
    parser.add_argument('--model_path', type=str, default='',
                        help='预训练模型路径')

    # 检索参数
    parser.add_argument('--topk', type=int, default=3,
                        help='示例检索返回的结果数量')
    parser.add_argument('--retrieval_top_k', type=int, default=5,
                        help='检索时使用的Top-K值')

    # 检索选项
    parser.add_argument('--example_retrieval', action='store_true',
                        help='运行示例检索并可视化结果')

    # 评估参数
    parser.add_argument('--eval_topk_list', type=int, nargs='+', default=[1, 5, 10],
                        help='评估时使用的Top-K值列表')
    parser.add_argument('--eval_single_view', action='store_true', default=True,
                        help='评估单视图检索性能')
    parser.add_argument('--eval_multiple_views', action='store_true', default=True,
                        help='评估多视图检索性能')
    parser.add_argument('--eval_view_retrieval', action='store_true', default=True,
                        help='评估视图检索性能')

    args = parser.parse_args()

    # 打印参数设置
    print("=" * 50)
    print("参数配置:")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("=" * 50)

    # 检查路径
    if args.mode == 'extract' and not args.model_path:
        # 尝试查找默认模型
        default_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(default_model_path):
            print("使用默认模型: {}".format(default_model_path))
            args.model_path = default_model_path
        else:
            raise FileNotFoundError("需要提供模型路径 (--model_path)")

    main(args)