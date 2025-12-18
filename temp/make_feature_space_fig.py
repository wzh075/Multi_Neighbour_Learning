import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成球面上均匀分布的点（使用黄金螺旋采样确保正圆均匀分布）
num_points = 5000  # 减少点数量避免粘连
radius = 1.0       # 球壳半径

# 黄金螺旋采样生成均匀分布的球坐标
phi = np.pi * (3 - np.sqrt(5))  # 黄金角

# 生成点坐标
y = 1 - (np.arange(num_points) / (num_points - 1)) * 2
theta = phi * np.arange(num_points)
r = np.sqrt(1 - y**2)
x = r * np.cos(theta)
z = r * np.sin(theta)

# 缩放至指定半径
x = radius * x
y = radius * y
z = radius * z

# 创建3D绘图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 使用HSL颜色空间实现低饱和度彩色效果，基于球坐标的角度
import matplotlib.colors as colors
colors_azimuth = np.arctan2(y, x)  # 计算点的方位角
colors_azimuth = (colors_azimuth + np.pi) / (2*np.pi)  # 归一化到0-1范围

# 生成低饱和度的HSL颜色
# 创建HSV颜色数组，饱和度设为0.5
hsv_colors = np.zeros((num_points, 3))
hsv_colors[:, 0] = colors_azimuth  # 色相
hsv_colors[:, 1] = 0.5             # 饱和度（降低至0.5）
hsv_colors[:, 2] = 0.8             # 亮度

# 转换为RGB颜色
rgb_colors = colors.hsv_to_rgb(hsv_colors)

# 绘制彩色球壳点云
scatter = ax.scatter(x, y, z, c=rgb_colors, s=8, alpha=0.8, marker='o', edgecolors='face')

# 设置坐标轴范围和标题
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_zlim(-1.1, 1.1)
ax.set_title('Feature Space on Spherical Shell', fontsize=16, color='black')

# 隐藏坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 隐藏所有背景网格和坐标轴
ax.xaxis.pane.set_visible(False)
ax.yaxis.pane.set_visible(False)
ax.zaxis.pane.set_visible(False)

# 隐藏坐标轴刻度线和标签
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 隐藏坐标轴线条
ax.xaxis.line.set_color('none')
ax.yaxis.line.set_color('none')
ax.zaxis.line.set_color('none')

# 保存图片
output_file = 'spherical_feature_space.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'图片已保存到: {output_file}')

# 显示图片（注释掉如果在没有GUI的环境下运行）
plt.show()