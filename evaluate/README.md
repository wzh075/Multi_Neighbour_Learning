# 多视图检索系统 - 验证模块

## 目录结构

```
evaluate/
├── evaluate.py        # 验证主程序，实现验证逻辑
├── evaluate.ps1       # Windows PowerShell 执行脚本
└── README.md          # 说明文档
```

## 功能说明

该验证模块用于评估多视图检索模型的性能，包含三种不同的检索任务：

1. **单视点组输入检索object**：使用单个视点组图像作为输入，检索相同的对象
2. **多视点组输入检索object**：使用多个视点组图像作为输入，检索相同的对象
3. **视点组之间互检索**：将输入视点组编码为视点特征，检索相似的视点组

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- torchvision
- numpy
- h5py
- tqdm
- argparse

## 使用方法

### 1. 准备工作

在运行验证之前，需要准备以下文件：

- **训练好的模型文件**（.pth格式）：包含模型权重
- **特征数据库**（.h5格式）：使用 `extract.py` 生成的特征数据库
- **验证数据集**：遵循 `root-类别-object-view-images` 的目录结构

### 2. 执行验证

#### 使用 Windows PowerShell 脚本（推荐）

```powershell
# 基本用法
.\evaluate.ps1 --model_path <模型文件路径> --feature_db_path <特征数据库路径>

# 完整示例
.\evaluate.ps1 `
    --root_dir "E:\Dataset\ModelNet40_Neighbour_view4_1.0" `
    --split "val" `
    --num_views 3 `
    --num_images_per_view 5 `
    --image_size 224 `
    --model_path "E:\Models\multi_view_model.pth" `
    --feat_dim 512 `
    --num_classes 40 `
    --feature_db_path "E:\Features\modelnet40_features.h5" `
    --output_dir ".\results"
```

#### 直接使用 Python 命令

```bash
# 基本用法
python evaluate.py --model_path <模型文件路径> --feature_db_path <特征数据库路径>

# 完整示例
python evaluate.py \
    --root_dir "E:\Dataset\ModelNet40_Neighbour_view4_1.0" \
    --split "val" \
    --num_views 3 \
    --num_images_per_view 5 \
    --image_size 224 \
    --model_path "E:\Models\multi_view_model.pth" \
    --feat_dim 512 \
    --num_classes 40 \
    --feature_db_path "E:\Features\modelnet40_features.h5" \
    --output_dir ".\results"
```

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `root_dir` | string | `E:\Dataset\ModelNet40_Neighbour_view4_1.0` | 数据集根目录 |
| `split` | string | `val` | 数据集分割（train/val/test） |
| `num_views` | int | `3` | 训练时设置的视点组数量 |
| `num_images_per_view` | int | `5` | 每个视点组的图像数量 |
| `image_size` | int | `224` | 输入图像尺寸 |
| `model_path` | string | 必填 | 训练得到的模型pth文件路径 |
| `feat_dim` | int | `512` | 特征维度大小 |
| `num_classes` | int | `40` | 类别数量 |
| `feature_db_path` | string | 必填 | 特征数据库路径 |
| `output_dir` | string | `./results` | 结果输出目录 |

## 验证流程

1. **加载模型**：从指定路径加载训练好的模型权重
2. **打开特征数据库**：加载预先生成的特征数据库
3. **数据预处理**：对输入图像进行 resize、normalize 等预处理
4. **创建验证数据集**：加载验证数据
5. **执行验证任务**：
   - 单视点组输入检索object
   - 多视点组输入检索object
   - 视点组之间互检索
6. **计算评估指标**：统计 top1、top5、top10 的 object 召回率和同 class 样本召回率
7. **保存结果**：将验证结果保存为 JSON 格式

## 评估指标

验证过程中会计算以下评估指标：

- **Object Recall**：检索到相同对象的召回率
  - Top-1 Object Recall
  - Top-5 Object Recall
  - Top-10 Object Recall

- **Class Recall**：检索到相同类别样本的召回率
  - Top-1 Class Recall
  - Top-5 Class Recall
  - Top-10 Class Recall

## 结果说明

验证结果会保存为 JSON 格式，文件路径为 `output_dir/evaluation_results.json`。

结果文件格式示例：

```json
{
  "single_view_retrieval": {
    "object_recall": {
      "1": 0.85,
      "5": 0.92,
      "10": 0.95
    },
    "class_recall": {
      "1": 0.95,
      "5": 0.98,
      "10": 0.99
    }
  },
  "multi_view_retrieval": {
    "object_recall": {
      "1": 0.90,
      "5": 0.96,
      "10": 0.98
    },
    "class_recall": {
      "1": 0.97,
      "5": 0.99,
      "10": 0.995
    }
  },
  "view_cross_retrieval": {
    "object_recall": {
      "1": 0.82,
      "5": 0.91,
      "10": 0.94
    },
    "class_recall": {
      "1": 0.94,
      "5": 0.98,
      "10": 0.99
    }
  }
}
```

## 数据集结构要求

验证数据集需要遵循以下目录结构：

```
root_dir/
├── class1/
│   ├── object1/
│   │   ├── view1/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── view2/
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── object2/
│   │   └── ...
│   └── ...
├── class2/
│   └── ...
└── ...
```

## 特征数据库结构

特征数据库需要使用 `extract.py` 生成，包含以下结构：

```
database.h5/
├── object1/
│   ├── obj_feat          # 对象特征
│   ├── view_features     # 视图特征列表
│   └── ...
├── object2/
│   └── ...
└── ...
```

## 注意事项

1. **模型参数一致性**：验证时使用的参数（如 `feat_dim`、`num_classes` 等）必须与训练时保持一致
2. **特征维度匹配**：`feat_dim` 参数必须与模型训练时的特征维度一致（默认为512）
3. **数据预处理**：验证时的图像预处理与训练时保持一致
4. **内存占用**：特征数据库较大时可能占用较多内存，建议在足够内存的环境中运行
5. **GPU加速**：如果有GPU，会自动使用GPU进行加速

## 常见问题

### 1. 模型加载失败

**错误信息**：`模型文件不存在` 或 `加载模型权重失败`

**解决方法**：
- 检查模型文件路径是否正确
- 确保模型文件格式正确（.pth格式）
- 检查模型参数是否与训练时一致

### 2. 特征数据库加载失败

**错误信息**：`特征数据库文件不存在` 或 `HDF5 错误`

**解决方法**：
- 检查特征数据库路径是否正确
- 确保使用 `extract.py` 正确生成了特征数据库

### 3. 数据集加载失败

**错误信息**：`数据集路径不存在` 或 `无法找到类别/对象`

**解决方法**：
- 检查数据集路径是否正确
- 确保数据集遵循要求的目录结构
- 检查 `num_views` 和 `num_images_per_view` 参数是否与数据集一致

### 4. 内存不足

**错误信息**：`Out of memory`

**解决方法**：
- 减少 `num_images_per_view` 参数
- 在内存更大的环境中运行
- 考虑使用更小的特征维度（需要重新训练模型）
