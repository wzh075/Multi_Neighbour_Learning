#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm  # 新增导入

def train_model(model, dataset, criterion, epochs=100, batch_size=32, lr=1e-3, save_dir='checkpoints', device=None):
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
    
    返回:
    - 训练好的模型
    """
    # 自动检测设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    print("训练设备: {}, 可用GPU数量: {}".format(device, num_gpus))
    
    # 如果使用多GPU，确保模型是并行模式
    if num_gpus > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    
    model = model.to(device)
    model.train()
    
    # 注释掉TensorBoard相关代码，因为环境不支持
    # writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard_logs'))
    
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
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # 初始化损失统计
        epoch_loss = 0.0
        epoch_infonce_loss = 0.0
        epoch_view_sim_loss = 0.0
        epoch_global_consistency_loss = 0.0
        epoch_obj_cluster_loss = 0.0
        
        start_time = time.time()
        zero_infonce_batches = 0  # 记录InfoNCE损失为0的批次数量
        
        # 使用tqdm创建进度条
        with tqdm(total=len(dataloader), desc="Epoch {}/{}".format(epoch+1, epochs),
                  bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # 确保张量连续并转移到设备
                images = batch['images'].contiguous().to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                obj_ids = batch['obj_ids']
                
                # 由于每个对象都有三个固定视点，且我们的损失函数已经能正确处理正样本对，无需额外检查
                
                # 分离三个视点组
                view1 = images[:, 0]
                view2 = images[:, 1]
                view3 = images[:, 2]
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(view1, view2, view3)
                
                # 计算损失
                # 确保模型输出中包含obj_feat，如果没有则使用global_feat作为替代
                obj_feat = outputs.get('obj_feat', outputs['global_feat'])
                # 使用labels作为class_labels
                loss_dict = criterion(
                    outputs['view_feats'], 
                    outputs['mid_feat'], 
                    outputs['global_feat'],
                    batch['obj_ids'],
                    obj_feat=obj_feat,
                    class_labels=labels
                )
                
                # 计算总损失
                loss = loss_dict['infonce_loss'] + loss_dict['view_similarity_loss'] + \
                       loss_dict['global_consistency_loss'] + loss_dict['obj_cluster_loss']
                
                # 检查InfoNCE损失是否接近0
                if loss_dict['infonce_loss'].item() < 1e-6:
                    zero_infonce_batches += 1
                    tqdm.write("警告: 批次 {} 的InfoNCE损失接近0: {:.8f}".format(batch_idx+1, loss_dict['infonce_loss'].item()))
                
                # 反向传播总损失
                loss.backward()
                
                # 添加梯度裁剪，提高训练稳定性
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 累计损失
                epoch_loss += loss.item()
                epoch_infonce_loss += loss_dict['infonce_loss'].item()
                epoch_view_sim_loss += loss_dict['view_similarity_loss'].item()
                epoch_global_consistency_loss += loss_dict['global_consistency_loss'].item()
                epoch_obj_cluster_loss += loss_dict['obj_cluster_loss'].item()
                
                # 注释掉TensorBoard日志记录，因为环境不支持
                # global_step = epoch * len(dataloader) + batch_idx
                # writer.add_scalar('Batch/Total_Loss', loss.item(), global_step)
                # writer.add_scalar('Batch/InfoNCE_Loss', loss_dict['infonce_loss'].item(), global_step)
                # writer.add_scalar('Batch/Mid_Global_Loss', loss_dict['mid_global_loss'].item(), global_step)
                # writer.add_scalar('Batch/Mid_Obj_Loss', loss_dict['mid_obj_loss'].item(), global_step)
                # writer.add_scalar('Batch/Zero_InfoNCE_Batches', zero_infonce_batches, global_step)
                
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']
                
                # 更新进度条显示的损失信息
                pbar.set_postfix({
                    'Loss': '{:.4f}'.format(loss.item()),
                    'InfoNCE_Loss': '{:.4f}'.format(loss_dict['infonce_loss'].item()),
                    'View_Sim_Loss': '{:.4f}'.format(loss_dict['view_similarity_loss'].item()),
                    'Global_Cons_Loss': '{:.4f}'.format(loss_dict['global_consistency_loss'].item()),
                    'Obj_Cluster_Loss': '{:.4f}'.format(loss_dict['obj_cluster_loss'].item()),
                    'LR': '{:.6f}'.format(current_lr)
                })
                pbar.update(1)
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_infonce_loss = epoch_infonce_loss / len(dataloader)
        avg_epoch_view_sim_loss = epoch_view_sim_loss / len(dataloader)
        avg_epoch_global_consistency_loss = epoch_global_consistency_loss / len(dataloader)
        avg_epoch_obj_cluster_loss = epoch_obj_cluster_loss / len(dataloader)
        
        # 每个epoch结束时的数据总结
        epoch_time = time.time() - start_time
        print("\n" + "="*80)
        print("Epoch {} 训练总结:".format(epoch+1))
        print("  训练时间: {:.2f} 秒".format(epoch_time))
        print("  平均总损失: {:.4f}".format(avg_epoch_loss))
        print("  平均InfoNCE损失: {:.4f}".format(avg_epoch_infonce_loss))
        print("  平均View_Sim损失: {:.4f}".format(avg_epoch_view_sim_loss))
        print("  平均Global_Cons损失: {:.4f}".format(avg_epoch_global_consistency_loss))
        print("  平均Obj_Cluster损失: {:.4f}".format(avg_epoch_obj_cluster_loss))
        
        # 打印零InfoNCE损失批次统计
        if zero_infonce_batches > 0:
            zero_percent = (zero_infonce_batches / len(dataloader)) * 100
            print("  零InfoNCE损失批次: {} ({:.2f}%)".format(zero_infonce_batches, zero_percent))
            if zero_percent > 50:
                print("  警告: 超过50%的批次InfoNCE损失为0，可能需要调整采样策略或批次大小")
        print("="*80 + "\n")
        
        # 学习率调整
        scheduler.step(avg_epoch_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 注释掉TensorBoard日志记录，因为环境不支持
        # writer.add_scalar('Epoch/Average_Loss', avg_epoch_loss, epoch)
        # writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
        # writer.add_scalar('Epoch/Time', epoch_time, epoch)
        
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
    
    # 注释掉TensorBoard写入器关闭，因为环境不支持
    # writer.close()
    
    return model