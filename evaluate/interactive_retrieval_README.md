# 交互式多视图检索脚本

## 功能介绍

本脚本允许用户通过命令行界面交互式地进行多视图检索。用户可以：

1. 选择数据集train子集中的一个类别
2. 选择该类别下的一个obj_id
3. 查看该object在三种检索模式下返回的前十个结果的class与obj_id
4. 反复进行查询操作

## 三种检索模式

1. **单视点组检索**：仅使用第一个视点组的图像进行检索
2. **多视点组检索**：使用所有视点组的图像进行检索
3. **视点组互检索**：使用第一个视点组的图像在视点数据库中进行检索

## 使用方法

### 基本命令

```bash
python interactive_retrieval.py --model_path <模型路径> --feature_db_path <特征数据库路径>
```

### 参数说明

- `--model_path`: 训练得到的模型pth文件路径（必需）
- `--feature_db_path`: 特征数据库路径（必需）
- `--root_dir`: 数据集根目录（默认：`/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0`）
- `--split`: 数据集分割（默认：`train`）
- `--num_views`: 训练时设置的视点组数量（默认：3）
- `--num_images_per_view`: 每个视点组的图像数量（默认：5）
- `--image_size`: 输入图像尺寸（默认：224）
- `--feat_dim`: 特征维度大小（默认：512）
- `--num_classes`: 类别数量（默认：40）

### 交互式操作流程

1. 脚本运行后，会显示所有可用的类别列表
2. 用户输入类别编号选择一个类别
3. 脚本会显示该类别下的所有obj_id列表
4. 用户输入obj_id编号选择一个对象
5. 脚本会使用三种检索模式检索该对象，并显示每种模式下的前10个结果
6. 用户可以选择继续查询或退出脚本

## 示例

```bash
python interactive_retrieval.py --model_path ./models/model_best.pth --feature_db_path ./feature_db
```

运行后，界面示例：

```
使用设备: cuda
训练数据集加载完成: 4000个样本

可用的类别:
1. airplane
2. bathtub
3. bed
...
40. xbox

请选择类别编号: 1

类别 'airplane' 下的obj_id:
1. 1
2. 2
3. 3
...

请选择obj_id编号: 5

已选择对象: 类别 = airplane, obj_id = 5

=== 单视点组检索结果 ===
1. class: airplane, obj_id: 5
2. class: airplane, obj_id: 12
3. class: airplane, obj_id: 3
...

=== 多视点组检索结果 ===
1. class: airplane, obj_id: 5
2. class: airplane, obj_id: 3
3. class: airplane, obj_id: 12
...

=== 视点组互检索结果 ===
1. class: airplane, obj_id: 5
2. class: airplane, obj_id: 8
3. class: airplane, obj_id: 12
...

是否继续查询? (y/n): y
```

## 注意事项

1. 请确保提供的模型路径和特征数据库路径是正确的
2. 脚本默认使用CUDA设备，如果没有GPU，会自动切换到CPU
3. 数据集路径可能需要根据您的系统进行调整
4. 确保您的环境中安装了所有必需的依赖项
