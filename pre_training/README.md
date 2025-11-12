# MVCNN预训练模块

本模块用于对多视图卷积神经网络(MVCNN)进行预训练，使用类别分类任务作为预训练任务，提高特征提取器的性能。

## 目录结构

```
pre_training/
├── pre_train_model.py      # 预训练MVCNN模型定义
├── pre_train_dataloader.py # 预训练数据集和数据加载器
├── pre_train.py            # 预训练主脚本
├── transfer_weights.py     # 权重迁移脚本
├── run_pre_train.sh        # 预训练启动脚本
└── README.md               # 本说明文件
```

## 预训练模型架构

`PreMVCNN`模型基于ResNet-18构建，包含以下组件：

1. **特征提取器**：基于ResNet-18的主干网络，移除最后的全连接层
2. **特征投影层**：将512维特征映射到指定维度，并应用LayerNorm优化特征分布
3. **分类头**：用于类别分类任务的全连接层

## 使用方法

### 1. 运行预训练

```bash
cd /Users/bytedance/PycharmProjects/Multi_Neighbour_Learning/pre_training
chmod +x run_pre_train.sh
./run_pre_train.sh
```

或者直接运行Python脚本：

```bash
python pre_train.py \
    --root_dir "你的数据集路径" \
    --batch_size 32 \
    --image_size 224 \
    --num_epochs 100 \
    --checkpoint_dir "../pre_checkpoints"
```

### 2. 权重迁移

预训练完成后，可以将权重迁移到主模型中：

```bash
python transfer_weights.py \
    --pre_trained_path "../pre_checkpoints/best_model.pth" \
    --output_path "../pre_checkpoints/transferred_encoder.pth" \
    --create_full_model \
    --full_model_output "../pre_checkpoints/pretrained_main_model.pth"
```

## 参数说明

### 预训练参数

- `--root_dir`: 数据集根目录
- `--batch_size`: 批量大小，默认32
- `--image_size`: 输入图像尺寸，默认224
- `--num_epochs`: 训练轮数，默认100
- `--learning_rate`: 学习率，默认1e-4
- `--weight_decay`: 权重衰减，默认1e-4
- `--checkpoint_dir`: 检查点保存目录，默认`../pre_checkpoints`
- `--log_dir`: 日志保存目录，默认`../logs/pre_train`

### 权重迁移参数

- `--pre_trained_path`: 预训练模型权重路径
- `--output_path`: 迁移后的权重保存路径
- `--create_full_model`: 是否创建完整的主模型
- `--full_model_output`: 完整主模型保存路径

## 数据格式要求

数据集应该按照以下结构组织：

```
dataset_root/
├── train/
│   ├── class1/
│   │   ├── object1/
│   │   │   ├── view1/
│   │   │   │   ├── img1.png
│   │   │   │   ├── img2.png
│   │   │   │   └── ...
│   │   │   ├── view2/
│   │   │   └── view3/
│   │   ├── object2/
│   │   └── ...
│   └── class2/
└── val/
    └── ... (与train结构相同)
```

## 预训练流程

1. 加载数据集并创建数据加载器
2. 初始化PreMVCNN模型
3. 执行训练循环，每个epoch包括：
   - 训练阶段
   - 验证阶段
   - 学习率调整
   - 保存检查点
4. 训练完成后，保存最佳模型
5. 使用transfer_weights.py将权重迁移到主模型

## 注意事项

1. 确保数据集路径正确，包含train和val子目录
2. 预训练权重将保存在`../pre_checkpoints`目录下
3. 迁移后的权重可以直接用于主模型的特征提取
4. 建议在迁移权重后，对主模型进行微调以获得最佳性能