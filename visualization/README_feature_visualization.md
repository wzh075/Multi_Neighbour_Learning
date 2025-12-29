# 特征空间可视化模块使用说明

本模块提供了强大的特征空间可视化功能，支持将HDF5格式存储的高维特征数据降维并可视化，帮助分析模型提取的特征分布情况。

## 功能特点

- 支持多种降维方法：t-SNE、PCA、UMAP
- 支持2D和3D可视化
- 按类别自动着色，直观显示不同类别特征分布
- 可控制每个类别显示的对象数量，避免可视化过于拥挤
- 支持多种特征类型：对象特征、全局特征、视图特征
- 自动处理NaN/Inf值，确保可视化质量
- 提供命令行接口和Python API两种使用方式

## 依赖安装

在使用前，请确保安装以下依赖：

```bash
pip install numpy matplotlib scikit-learn umap-learn seaborn h5py
```

## 使用方法

### 方法一：使用shell脚本（推荐）

我们提供了一个便捷的shell脚本，支持丰富的命令行参数：

```bash
# 进入可视化目录
cd visualization

# 生成所有可视化结果（t-SNE、PCA、UMAP的2D和3D可视化）
./visualize.sh

# 使用t-SNE方法生成2D可视化
./visualize.sh --method tsne --components 2

# 显示图像而不保存
./visualize.sh --method tsne --show

# 可视化全局特征，限制每个类别最多20个对象
./visualize.sh --feat_type global_features --num_objects 20

# 使用自定义数据库和输出目录
./visualize.sh --feature_db ../../custom_features.h5 --output_dir ../../my_visuals

# 查看帮助信息
./visualize.sh --help
```

### 方法二：直接使用Python脚本

**基本使用示例：**

```bash
# 进入可视化目录
cd visualization

# 生成所有可视化结果
python feature_visualization.py --feature_db ../../features/feature_db.h5 --output_dir ../../visualizations

# 使用t-SNE方法生成2D可视化
python feature_visualization.py --feature_db ../../features/feature_db.h5 --method tsne --n_components 2 --output_dir ../../visualizations

# 显示图像而不保存
python feature_visualization.py --feature_db ../../features/feature_db.h5 --method tsne --show

# 可视化全局特征，限制每个类别最多20个对象
python feature_visualization.py --feature_db ../../features/feature_db.h5 --feat_type global_features --max_objects_per_class 20
```

### 方法三：Python API使用

```python
from visualization.feature_visualization import FeatureVisualizer

# 创建可视化器实例
visualizer = FeatureVisualizer('../../features/feature_db.h5')

# 加载特征（每个类别最多50个对象）
visualizer.load_features(feat_type='obj_feat', max_objects_per_class=50)

# 使用t-SNE降维到2D
visualizer.reduce_dimension(method='tsne', n_components=2)

# 保存可视化结果
visualizer.plot_2d(output_path='../../visualizations/tsne_2d.png', title='t-SNE特征空间可视化')

# 生成所有类型的可视化
visualizer.generate_all_visualizations(output_dir='../../visualizations', max_objects_per_class=50)
```

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--feature_db` | str | `../../features/feature_db.h5` | 特征数据库文件路径 |
| `--output_dir` | str | `../../visualizations` | 可视化结果保存目录 |
| `--method` | str | `all` | 降维方法，可选：`tsne`、`pca`、`umap`、`all` |
| `--n_components` | int | `2` | 降维后的维度，可选：`2`、`3` |
| `--max_objects_per_class` | int | `50` | 每个类别最多可视化的对象数量 |
| `--feat_type` | str | `obj_feat` | 要可视化的特征类型，可选：`obj_feat`、`global_features`、`view_features` |
| `--show` | bool | `False` | 是否显示图像而不是保存 |

## 可视化结果说明

生成的可视化图像中：
- 不同类别的对象会显示为不同颜色
- 同一类别的对象会使用相同颜色
- 点的大小可以通过参数调整
- 图例显示每个颜色对应的类别名称

## 注意事项

1. 确保特征数据库文件存在且格式正确
2. 对于大规模数据集，建议设置较小的`max_objects_per_class`以提高可视化效果
3. 3D可视化在保存为静态图片时可能无法完全展示效果，建议使用`--show`参数在交互式环境中查看
4. 如果类别信息提取不正确，请根据实际数据集修改`load_features`方法中的类别提取逻辑

## 扩展与自定义

如果需要进一步自定义可视化效果，可以修改以下部分：

1. 在`load_features`方法中调整类别信息的提取逻辑
2. 在`plot_2d`和`plot_3d`方法中修改点的大小、颜色、透明度等参数
3. 添加新的降维方法或可视化类型

## 测试功能

我们还提供了一个测试脚本，可以自动生成示例特征数据库并进行可视化：

```bash
# 进入可视化目录
cd visualization

# 生成示例数据库并测试
python test_visualization.py --generate_sample

# 使用现有数据库测试
python test_visualization.py
```

## 故障排除

- **特征加载失败**：检查特征数据库路径是否正确，确保HDF5文件格式正确
- **类别信息不正确**：修改`load_features`方法中的类别提取逻辑
- **可视化过于拥挤**：减小`max_objects_per_class`参数值
- **内存不足**：处理大规模数据集时，减小`max_objects_per_class`或分批次处理
- **脚本执行权限错误**：运行 `chmod +x visualize.sh` 确保脚本有执行权限

# 特征可视化说明

本目录包含多个特征可视化脚本，用于可视化特征数据库中的不同特征类型。

## 脚本说明

### 1. feature_visualization.py
- **功能**: 通用特征可视化脚本，支持多种降维方法
- **特点**: 
  - 支持t-SNE、PCA、UMAP等多种降维方法
  - 支持2D和3D可视化
  - 按类别自动着色
  - 可控制每个类别显示的对象数量

### 2. object_feature_visualization.py
- **功能**: 可视化object级别的特征，将每个对象作为一个点进行可视化
- **特点**: 
  - 支持不同特征类型 (obj_feat, global_features, view_features)
  - 不同颜色表示不同类别
  - 生成3D、2D PCA和2D t-SNE图
  - 支持中文标签显示

### 3. view_feature_visualization.py
- **功能**: 可视化view级别的特征，将每个view作为一个点进行可视化
- **特点**:
  - 专门用于多视角特征分析
  - 不同颜色表示不同对象
  - 生成3D、2D PCA和2D t-SNE图
  - 支持中文标签显示
  - 可限制每个对象的最大view数量以控制数据量

## 运行脚本

### 使用Python直接运行

```bash
# 通用特征可视化
python feature_visualization.py --feature_db ../features/feature_db.h5 --output_dir ../visualizations

# Object级别特征可视化
python object_feature_visualization.py --feature_db_path ../features/feature_db.h5 --feat_type obj_feat --output_dir ../visualizations --max_samples 500

# View级别特征可视化
python view_feature_visualization.py --feature_db_path ../features/feature_db.h5 --output_dir ../visualizations --max_samples 500 --max_views_per_obj 5
```

### 使用PowerShell脚本

```powershell
# Object级别特征可视化
.\visualize_obj_features.ps1 --feature_db_path "../features/feature_db.h5" --feat_type "obj_feat" --output_dir "../visualizations" --max_samples 500

# View级别特征可视化
.\visualize_view_features.ps1 --feature_db_path "../features/feature_db.h5" --output_dir "../visualizations" --max_samples 500 --max_views_per_obj 5
```

### 使用Bash脚本

```bash
# Object级别特征可视化
./visualize_obj_features.sh --feature_db_path "../features/feature_db.h5" --feat_type "obj_feat" --output_dir "../visualizations" --max_samples 500

# View级别特征可视化
./visualize_view_features.sh --feature_db_path "../features/feature_db.h5" --output_dir "../visualizations" --max_samples 500 --max_views_per_obj 5
```

## 参数说明

- `--feature_db` 或 `--feature_db_path`: 特征数据库路径 (.h5文件)
- `--feat_type`: 特征类型 (仅object_feature_visualization.py使用)
  - `obj_feat`: 对象特征
  - `global_features`: 全局特征
  - `view_features`: 视角特征
- `--output_dir`: 输出目录路径
- `--max_samples`: 最大采样数量（如果指定，将从特征中随机采样）
- `--max_views_per_obj`: 每个对象的最大view数量（仅view_feature_visualization.py使用）

## 输出文件

脚本会生成以下可视化文件：

- `{type}_3d_scatter.png`: 3D散点图
- `{type}_pca_2d.png`: 2D PCA降维图
- `{type}_tsne_2d.png`: 2D t-SNE降维图

其中 `{type}` 为特征类型名称。

## 特性

- **中文支持**: 自动检测并使用中文字体，支持中文标签显示
- **类别着色**: 不同类别/对象使用不同颜色显示
- **多种降维方法**: 支持PCA和t-SNE降维
- **统计分析**: 提供特征分布的统计信息
- **灵活配置**: 支持多种参数配置以适应不同需求