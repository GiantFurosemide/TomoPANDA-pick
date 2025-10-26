#!/usr/bin/env python3
"""
io_mrc 模块使用示例

展示如何使用 Dataset + HDF5 方案处理大量 MRC 文件
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.io_mrc import convert_mrcs_to_hdf5, HDF5VolumeDataset, create_sample_mrc_files
import torch
from torch.utils.data import DataLoader
import numpy as np


def main():
    """主函数 - 演示完整的使用流程"""
    print("=== MRC 文件处理示例 ===\n")
    
    # 1. 创建示例 MRC 文件
    print("1. 创建示例 MRC 文件...")
    mrc_files = create_sample_mrc_files("example_data", 20)
    print(f"   创建了 {len(mrc_files)} 个 MRC 文件")
    
    # 2. 转换为 HDF5 格式
    print("\n2. 转换为 HDF5 格式...")
    hdf5_path = "example_volumes.h5"
    convert_mrcs_to_hdf5(mrc_files, hdf5_path)
    
    # 3. 创建 PyTorch Dataset
    print("\n3. 创建 PyTorch Dataset...")
    dataset = HDF5VolumeDataset(hdf5_path, cache_size=10)
    print(f"   数据集大小: {len(dataset)}")
    
    # 4. 查看数据信息
    print("\n4. 查看数据信息...")
    metadata = dataset.get_metadata()
    print(f"   元数据数量: {len(metadata)}")
    print(f"   第一个文件: {metadata[0]['file_path']}")
    print(f"   第一个文件尺寸: {metadata[0]['shape']}")
    
    # 5. 测试数据访问
    print("\n5. 测试数据访问...")
    sample_data = dataset[0]
    print(f"   第一个样本形状: {sample_data.shape}")
    print(f"   第一个样本数据类型: {sample_data.dtype}")
    print(f"   第一个样本数值范围: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
    
    # 6. 测试缓存功能
    print("\n6. 测试缓存功能...")
    print(f"   缓存大小: {len(dataset.cache)}")
    
    # 访问相同数据（应该从缓存获取）
    cached_data = dataset[0]
    print(f"   缓存命中: {torch.equal(sample_data, cached_data)}")
    
    # 7. 测试数据变换
    print("\n7. 测试数据变换...")
    def normalize_transform(data):
        """归一化变换"""
        return (data - data.mean()) / data.std()
    
    dataset_with_transform = HDF5VolumeDataset(hdf5_path, transform=normalize_transform)
    normalized_data = dataset_with_transform[0]
    print(f"   归一化后均值: {normalized_data.mean():.6f}")
    print(f"   归一化后标准差: {normalized_data.std():.6f}")
    
    # 8. 测试 PyTorch DataLoader
    print("\n8. 测试 PyTorch DataLoader...")
    
    # 创建相同尺寸的数据集用于批处理
    same_size_files = []
    for i in range(10):
        import mrcfile
        mrc_file = f"example_data/same_size_{i:03d}.mrc"
        data = np.random.rand(64, 64, 64).astype(np.float32)
        with mrcfile.new(mrc_file) as mrc:
            mrc.set_data(data)
            mrc.update_header_from_data()
        same_size_files.append(mrc_file)
    
    # 转换为 HDF5
    convert_mrcs_to_hdf5(same_size_files, "example_same_size.h5")
    
    # 创建相同尺寸的数据集
    same_size_dataset = HDF5VolumeDataset("example_same_size.h5")
    
    # 创建 DataLoader
    dataloader = DataLoader(
        same_size_dataset, 
        batch_size=4, 
        shuffle=True,
        num_workers=0
    )
    
    print("   批处理测试:")
    for i, batch in enumerate(dataloader):
        print(f"     批次 {i+1}: 形状 {batch.shape}")
        if i >= 2:  # 只显示前3个批次
            break
    
    # 9. 清理
    print("\n9. 清理临时文件...")
    import shutil
    shutil.rmtree("example_data")
    os.remove("example_volumes.h5")
    os.remove("example_same_size.h5")
    print("   清理完成")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()
