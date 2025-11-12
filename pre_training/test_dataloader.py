import os
import sys
import torch
from pre_train_dataloader import get_pre_train_dataloaders


def test_dataloader(dataset_path):
    """
    测试数据加载器是否正常工作
    """
    print(f"\n====== 开始测试数据加载器 ======\n")
    print(f"数据集路径: {dataset_path}")
    
    try:
        # 获取数据加载器
        dataloaders = get_pre_train_dataloaders(
            root_dir=dataset_path,
            batch_size=8,  # 使用较小的批次大小进行测试
            image_size=224,
            num_workers=2,
            num_views=3,
            num_images_per_view=5
        )
        
        print(f"\n数据加载器创建成功!")
        print(f"训练集批次数量: {len(dataloaders['train'])}")
        print(f"验证集批次数量: {len(dataloaders['val'])}")
        print(f"类别数量: {len(dataloaders['class_to_idx'])}")
        print(f"类别列表: {list(dataloaders['class_to_idx'].keys())[:10]}...")
        
        # 尝试加载一个批次的数据
        print("\n尝试加载第一个训练批次...")
        for images, labels in dataloaders['train']:
            print(f"批次形状: 图像={images.shape}, 标签={labels.shape}")
            print(f"标签范围: {labels.min()} - {labels.max()}")
            print(f"标签样例: {labels[:5]}")
            print("\n数据加载成功！数据加载器工作正常。")
            break  # 只测试第一个批次
        
        # 尝试加载验证集数据
        print("\n尝试加载第一个验证批次...")
        for images, labels in dataloaders['val']:
            print(f"批次形状: 图像={images.shape}, 标签={labels.shape}")
            print("\n验证集数据加载成功！")
            break  # 只测试第一个批次
        
        print(f"\n====== 数据加载器测试成功 ======\n")
        return True
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n====== 数据加载器测试失败 ======\n")
        return False


if __name__ == "__main__":
    # 使用命令行参数或默认路径
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # 默认路径，可根据需要修改
        dataset_path = '/data1/Wuzhihe/Dataset/ModelNet40_Neighbour_view4_1.0'
    
    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集路径不存在: {dataset_path}")
        # 尝试使用当前目录作为测试
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"\n尝试使用当前目录作为测试: {current_dir}")
        dataset_path = current_dir
    
    # 运行测试
    test_dataloader(dataset_path)