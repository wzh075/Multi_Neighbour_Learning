#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pre_training.pre_train_model import PreMVCNN
from pre_training.pre_train_dataloader import get_pre_train_dataloaders

def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    """
    训练一个epoch
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1} - Training') as pbar:
        for batch_idx, batch in enumerate(dataloader):
            # 准备数据
            images = batch['images'].to(device)  # [batch, num_views, num_images, C, H, W]
            class_idx = batch['class_idx'].to(device)
            batch_size = images.size(0)
            
            # 调整图像维度，将多个视图的图像合并
            # 从 [batch, num_views, num_images, C, H, W] 转换为 [batch, num_views*num_images, C, H, W]
            batch_size, num_views, num_images, C, H, W = images.size()
            images = images.view(batch_size, num_views * num_images, C, H, W)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            logits = outputs['logits']
            
            # 计算损失
            loss = criterion(logits, class_idx)
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * batch_size
            _, preds = torch.max(logits, 1)
            running_corrects += torch.sum(preds == class_idx.data)
            total_samples += batch_size
            
            # 更新进度条
            pbar.set_postfix({
                'loss': running_loss / total_samples,
                'acc': running_corrects.double() / total_samples
            })
            pbar.update(1)
            
            # 记录到TensorBoard
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/batch_loss', loss.item(), global_step)
            writer.add_scalar('Train/batch_acc', torch.sum(preds == class_idx.data).double() / batch_size, global_step)
    
    # 计算整个epoch的统计
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    # 记录到TensorBoard
    writer.add_scalar('Train/epoch_loss', epoch_loss, epoch)
    writer.add_scalar('Train/epoch_acc', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, writer, epoch):
    """
    验证模型
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch+1} - Validation') as pbar:
            for batch in dataloader:
                # 准备数据
                images = batch['images'].to(device)
                class_idx = batch['class_idx'].to(device)
                batch_size = images.size(0)
                
                # 调整图像维度
                batch_size, num_views, num_images, C, H, W = images.size()
                images = images.view(batch_size, num_views * num_images, C, H, W)
                
                # 前向传播
                outputs = model(images)
                logits = outputs['logits']
                
                # 计算损失
                loss = criterion(logits, class_idx)
                
                # 统计
                running_loss += loss.item() * batch_size
                _, preds = torch.max(logits, 1)
                running_corrects += torch.sum(preds == class_idx.data)
                total_samples += batch_size
                
                # 收集预测和标签用于混淆矩阵
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(class_idx.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': running_loss / total_samples,
                    'acc': running_corrects.double() / total_samples
                })
                pbar.update(1)
    
    # 计算整个验证集的统计
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    # 记录到TensorBoard
    writer.add_scalar('Val/loss', epoch_loss, epoch)
    writer.add_scalar('Val/acc', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, best_acc, checkpoint_dir, is_best=False):
    """
    保存模型检查点
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    
    # 只保存最佳模型
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved to {best_model_path}")

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建必要的目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 获取数据加载器
    dataloaders = get_pre_train_dataloaders(
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        num_views=args.num_views,
        num_images_per_view=args.num_images_per_view
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    class_to_idx = dataloaders['class_to_idx']
    
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {class_to_idx}")
    
    # 初始化模型
    model = PreMVCNN(
        num_classes=num_classes,
        feat_dim=args.feat_dim
    )
    model = model.to(device)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True
    )
    
    # 创建TensorBoard记录器
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f'pre_train_{timestamp}'))
    
    # 加载检查点（如果有）
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            
            # 加载模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Loaded checkpoint at epoch {start_epoch}, best acc: {best_acc:.4f}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # 开始训练
    print("=" * 60)
    print(f"Starting pre-training for {args.num_epochs} epochs")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, writer, epoch
        )
        
        # 更新学习率
        scheduler.step(val_acc)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/learning_rate', current_lr, epoch)
        
        # 打印epoch摘要
        print(f"Epoch {epoch+1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Current LR: {current_lr:.6f}")
        
        # 保存检查点
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        # 只在验证准确率提升时保存特征提取器的权重（用于迁移）
        if is_best:
            if isinstance(model, nn.DataParallel):
                feature_extractor = model.module.get_feature_extractor()
            else:
                feature_extractor = model.get_feature_extractor()
            
            torch.save(
                feature_extractor.state_dict(),
                os.path.join(args.checkpoint_dir, 'best_feature_extractor.pth')
            )
            print(f"Best feature extractor saved to {os.path.join(args.checkpoint_dir, 'best_feature_extractor.pth')}")
        
        # 保存检查点（只保存最佳模型）
    save_checkpoint(model, optimizer, epoch, best_acc, args.checkpoint_dir, is_best)
        
    # 训练完成
    print("=" * 60)
    print(f"Pre-training completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print("=" * 60)
    
    # 保存最终模型
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'class_to_idx': class_to_idx
    }, final_model_path)
    
    print(f"Final model saved to {final_model_path}")
    writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description='MVCNN预训练脚本')
    
    # 数据参数
    parser.add_argument('--root_dir', type=str, default='/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0',
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--image_size', type=int, default=224,
                        help='图像尺寸')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作线程数')
    parser.add_argument('--num_views', type=int, default=3,
                        help='每个对象使用的视图数量')
    parser.add_argument('--num_images_per_view', type=int, default=5,
                        help='每个视图使用的图像数量')
    
    # 模型参数
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='特征维度')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='学习率衰减因子')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='学习率衰减耐心值')
    
    # 保存和恢复参数
    parser.add_argument('--checkpoint_dir', type=str, default='../pre_checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='../logs/pre_train',
                        help='日志保存目录')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"数据集路径: {args.root_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像大小: {args.image_size}")
    print(f"视图数量: {args.num_views}")
    print(f"每视图图像数: {args.num_images_per_view}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"检查点目录: {args.checkpoint_dir}")
    main(args)