#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm  # 进度条显示
from torch.utils.tensorboard import SummaryWriter  # TensorBoard支持
# 添加混合精度训练支持
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    print("警告: torch.cuda.amp不可用，将不使用混合精度训练")

def train_model(model, dataset, criterion, epochs=100, batch_size=32, lr=1e-3, save_dir='checkpoints', device=None, tensorboard_dir='../tensorboard_log/training'):
    """
    训练模型的主函数
    
    参数:
    - model: 要训练的模型
    - dataset: 训练数据集
    - criterion: 损失函数
    - epochs: 训练轮数
    - batch_size: 批大小
    - lr: 学习率
    - save_dir: 模型保存目录
    - device: 训练设备
    - tensorboard_dir: TensorBoard日志保存目录
    
    返回:
    - 训练好的模型
    """
    # 自动检测设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    print("训练设备: {}, 可用GPU数量: {}".format(device, num_gpus))
    
    # 为了提高InfoNCE的效果，我们需要足够大的batch_size
    # 保留原始batch_size，不进行过度的显存优化
    print(f"训练批大小: {batch_size}，将通过梯度累积模拟更大的批大小")
    
    # 如果使用多GPU，确保模型是并行模式
    if num_gpus > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    
    model = model.to(device)
    model.train()
    
    # 启用梯度累积以模拟更大的批大小
    # 增加累积步数以有效扩大batch_size
    accumulation_steps = 8  # 累积8步梯度再更新，有效batch_size = batch_size * 8
    print(f"启用梯度累积: {accumulation_steps} 步，有效批大小: {batch_size * accumulation_steps}")
    
    # 初始化混合精度训练scaler
    scaler = GradScaler() if AMP_AVAILABLE and device.type == 'cuda' else None
    
    # 创建数据加载器 - 根据GPU数量调整num_workers，使用常规的shuffle
    num_workers = min(8, 4 * num_gpus) if num_gpus > 0 else 4
    
    # 由于每个对象都恰好包含三个视点，使用普通的shuffle即可
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,    # 使用默认的shuffle功能
        num_workers=num_workers,
        pin_memory=True,  # 启用内存锁页，加速数据传输
        collate_fn=lambda batch: {
            'images': torch.stack([item['images'] for item in batch]).contiguous(),
            'labels': torch.tensor([item['label'] for item in batch]),
            'obj_ids': [item['obj_id'] for item in batch]  # 使用唯一的obj_id字符串列表
        }
    )
    
    # 优化器 - 调整权重衰减以改善特征分布
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)  # 减小权重衰减
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建TensorBoard记录器
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # 初始化损失统计
        epoch_loss = 0.0
        epoch_infonce_loss = 0.0
        epoch_view_sim_loss = 0.0
        epoch_global_consistency_loss = 0.0
        
        start_time = time.time()
        zero_infonce_batches = 0  # 记录InfoNCE损失为0的批次数量
        
        # 使用tqdm创建进度条
        with tqdm(total=len(dataloader), desc="Epoch {}/{} (训练)".format(epoch+1, epochs),
                  bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # 确保张量连续并转移到设备
                images = batch['images'].contiguous().to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                obj_ids = batch['obj_ids']
                
                # 分离三个视点组
                view1 = images[:, 0]
                view2 = images[:, 1]
                view3 = images[:, 2]
                
                # 将字符串obj_ids转换为数值型索引
                # 创建唯一的数值索引映射
                unique_obj_ids = list(set(obj_ids))
                obj_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(unique_obj_ids)}
                obj_indices = torch.tensor([obj_id_to_idx[obj_id] for obj_id in obj_ids], device=device)
                
                # 梯度累积：只在累积开始时清零梯度
                if (batch_idx + 1) % accumulation_steps == 1:
                    optimizer.zero_grad()
                
                # 使用混合精度训练
                if scaler is not None:
                    with autocast():
                        # 前向传播
                        outputs = model(view1, view2, view3)
                        
                        # 计算损失，使用原始特征而不是归一化后的特征计算损失
                        obj_feat = outputs.get('obj_feat', outputs['raw_global_feat'])
                        loss_dict = criterion(
                            outputs['view_feats'], 
                            outputs['raw_mid_feat'], 
                            outputs['raw_global_feat'],
                            obj_indices,  # 使用数值型索引而不是字符串
                            obj_feat=obj_feat,
                            class_labels=labels,
                            raw_mid_feat=outputs.get('raw_mid_feat', None),
                            raw_global_feat=outputs.get('raw_global_feat', None),
                            raw_obj_feat=outputs.get('raw_obj_feat', None)
                        )
                        
                        # 重新平衡损失权重，确保各损失项都有适当贡献
                        # 使用更合理的组合
                        loss = (loss_dict['infonce_loss'] * 2.0 + 
                                loss_dict['view_similarity_loss'] + 
                                loss_dict['global_consistency_loss'] + 
                                loss_dict.get('feat_mean_reg', 0.0)) / accumulation_steps
                        
                        # 添加调试信息，定期检查各损失项的值
                        if (batch_idx + 1) % 100 == 0:
                            global_step = epoch * len(dataloader) + batch_idx
                            print(f"\n批次 {batch_idx} 损失详情:")
                            print(f"  InfoNCE损失: {loss_dict['infonce_loss'].item():.6f}")
                            print(f"  视图相似度损失: {loss_dict['view_similarity_loss'].item():.6f}")
                            print(f"  全局一致性损失: {loss_dict['global_consistency_loss'].item():.6f}")
                            print(f"  特征正则化损失: {loss_dict.get('feat_mean_reg', 0.0).item():.6f}")
                            print(f"  总损失: {loss.item():.6f}")
                            
                            # 检查特征统计信息
                            if outputs.get('view_feats') is not None:
                                print(f"  视图特征均值: {outputs['view_feats'].mean().item():.6f}, 标准差: {outputs['view_feats'].std().item():.6f}")
                            if outputs.get('global_feat') is not None:
                                print(f"  全局特征均值: {outputs['global_feat'].mean().item():.6f}, 标准差: {outputs['global_feat'].std().item():.6f}")
                else:
                    # 常规训练
                    outputs = model(view1, view2, view3)
                    obj_feat = outputs.get('obj_feat', outputs['raw_global_feat'])
                    loss_dict = criterion(
                        outputs['view_feats'], 
                        outputs['raw_mid_feat'], 
                        outputs['raw_global_feat'],
                        obj_indices,  # 使用数值型索引而不是字符串
                        obj_feat=obj_feat,
                        class_labels=labels,
                        raw_mid_feat=outputs.get('raw_mid_feat', None),
                        raw_global_feat=outputs.get('raw_global_feat', None),
                        raw_obj_feat=outputs.get('raw_obj_feat', None)
                    )
                    
                    # 重新平衡损失权重，确保各损失项都有适当贡献
                    # 使用更合理的组合
                    loss = (loss_dict['infonce_loss'] * 2.0 + 
                            loss_dict['view_similarity_loss'] + 
                            loss_dict['global_consistency_loss'] + 
                            loss_dict.get('feat_mean_reg', 0.0)) / accumulation_steps
                    
                    # 添加调试信息，定期检查各损失项的值
                    if (batch_idx + 1) % 100 == 0:
                        global_step = epoch * len(dataloader) + batch_idx
                        print(f"\n批次 {batch_idx} 损失详情:")
                        print(f"  InfoNCE损失: {loss_dict['infonce_loss'].item():.6f}")
                        print(f"  视图相似度损失: {loss_dict['view_similarity_loss'].item():.6f}")
                        print(f"  全局一致性损失: {loss_dict['global_consistency_loss'].item():.6f}")
                        print(f"  特征正则化损失: {loss_dict.get('feat_mean_reg', 0.0).item():.6f}")
                        print(f"  总损失: {loss.item():.6f}")
                        
                        # 检查特征统计信息
                        if outputs.get('view_feats') is not None:
                            print(f"  视图特征均值: {outputs['view_feats'].mean().item():.6f}, 标准差: {outputs['view_feats'].std().item():.6f}")
                        if outputs.get('global_feat') is not None:
                            print(f"  全局特征均值: {outputs['global_feat'].mean().item():.6f}, 标准差: {outputs['global_feat'].std().item():.6f}")
                
                # 降低特征分布记录频率，减少内存占用
                if (batch_idx + 1) % 100 == 0:
                    global_step = epoch * len(dataloader) + batch_idx
                    writer.add_scalar('Features/view_feats_mean', outputs['view_feats'].mean().item(), global_step)
                    writer.add_scalar('Features/view_feats_std', outputs['view_feats'].std().item(), global_step)
                    writer.add_scalar('Features/mid_feat_mean', outputs['mid_feat'].mean().item(), global_step)
                    writer.add_scalar('Features/mid_feat_std', outputs['mid_feat'].std().item(), global_step)
                    writer.add_scalar('Features/global_feat_mean', outputs['global_feat'].mean().item(), global_step)
                    writer.add_scalar('Features/global_feat_std', outputs['global_feat'].std().item(), global_step)
                    # 添加原始全局特征的统计
                    writer.add_scalar('Features/raw_global_feat_mean', outputs['raw_global_feat'].mean().item(), global_step)
                    writer.add_scalar('Features/raw_global_feat_std', outputs['raw_global_feat'].std().item(), global_step)
                
                # 检查InfoNCE损失是否接近0
                if loss_dict['infonce_loss'].item() < 1e-6:
                    zero_infonce_batches += 1
                    
                # 已移除对象聚类损失，无需检查
                
                # 反向传播
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # 梯度裁剪
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    loss.backward()
                    # 梯度裁剪和参数更新
                    if (batch_idx + 1) % accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                # 安全累计损失，检查是否为有效数值
                if not torch.isinf(loss).any() and not torch.isnan(loss).any():
                    epoch_loss += loss.item()
                
                if not torch.isinf(loss_dict['infonce_loss']).any() and not torch.isnan(loss_dict['infonce_loss']).any():
                    epoch_infonce_loss += loss_dict['infonce_loss'].item()
                
                if not torch.isinf(loss_dict['view_similarity_loss']).any() and not torch.isnan(loss_dict['view_similarity_loss']).any():
                    epoch_view_sim_loss += loss_dict['view_similarity_loss'].item()
                
                if not torch.isinf(loss_dict['global_consistency_loss']).any() and not torch.isnan(loss_dict['global_consistency_loss']).any():
                    epoch_global_consistency_loss += loss_dict['global_consistency_loss'].item()
                
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 更新进度条显示，展示所有损失值
                pbar.set_postfix({
                    'Total': '{:.4f}'.format(loss.item()),
                    'InfoNCE': '{:.4f}'.format(loss_dict['infonce_loss'].item()),
                    'ViewSim': '{:.4f}'.format(loss_dict['view_similarity_loss'].item()),
                    'Global': '{:.4f}'.format(loss_dict['global_consistency_loss'].item()),
                    'LR': '{:.6f}'.format(current_lr)
                })
                
                # 清理所有中间变量以释放显存
                del outputs, obj_feat, loss_dict, loss, view1, view2, view3, images, labels
                # 强制垃圾回收
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                pbar.update(1)
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_infonce_loss = epoch_infonce_loss / len(dataloader)
        avg_epoch_view_sim_loss = epoch_view_sim_loss / len(dataloader)
        avg_epoch_global_consistency_loss = epoch_global_consistency_loss / len(dataloader)
        
        # 每个epoch结束时的数据总结
        epoch_time = time.time() - start_time
        
        # 记录epoch级别的指标到TensorBoard
        writer.add_scalar('Losses/total_loss', avg_epoch_loss, epoch)
        writer.add_scalar('Losses/infonce_loss', avg_epoch_infonce_loss, epoch)
        writer.add_scalar('Losses/view_similarity_loss', avg_epoch_view_sim_loss, epoch)
        writer.add_scalar('Losses/global_consistency_loss', avg_epoch_global_consistency_loss, epoch)
        writer.add_scalar('Metrics/learning_rate', current_lr, epoch)
        writer.add_scalar('Metrics/epoch_time', epoch_time, epoch)
        
        # 控制台输出总结信息，展示所有损失值
        print(f"\nEpoch {epoch+1}/{epochs} 训练完成")
        print(f"  训练时间: {epoch_time:.2f} 秒")
        print(f"  平均总损失: {avg_epoch_loss:.4f}")
        print(f"  InfoNCE损失: {avg_epoch_infonce_loss:.4f}")
        print(f"  视图相似度损失: {avg_epoch_view_sim_loss:.4f}")
        print(f"  全局一致性损失: {avg_epoch_global_consistency_loss:.4f}")
        
        # 添加调试信息，检查损失值是否异常小
        if avg_epoch_loss < 0.01:
            print(f"  警告: 总损失值异常小 ({avg_epoch_loss:.6f})，可能存在梯度消失或模型过拟合问题")
        if avg_epoch_infonce_loss < 0.001:
            print(f"  警告: InfoNCE损失值异常小 ({avg_epoch_infonce_loss:.6f})，可能特征区分度过低")
        
        # 零InfoNCE损失批次统计 - 只在超过阈值时打印警告
        if zero_infonce_batches > 0:
            zero_percent = (zero_infonce_batches / len(dataloader)) * 100
            if zero_percent > 50:
                print(f"  警告: 超过50%的批次InfoNCE损失为0 ({zero_percent:.2f}%)，建议调整采样策略或批次大小")
        
        # 记录到TensorBoard的额外信息已完成
        
        # 学习率调整
        scheduler.step(avg_epoch_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, save_path)
            print("最佳模型已保存至: {}".format(save_path))
    
    # 关闭TensorBoard记录器
    writer.close()
    print(f"TensorBoard日志已保存至: {tensorboard_dir}")
    
    return model