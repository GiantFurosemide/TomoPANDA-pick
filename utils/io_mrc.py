"""
MRC/MRCS 文件操作工具模块

提供基于 mrcfile 和 HDF5 的 MRC 文件处理功能，特别适用于大量小文件的深度学习训练。
"""

import h5py
import mrcfile
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable, Any
from pathlib import Path
import json


def convert_mrcs_to_hdf5(mrc_files: List[str], output_path: str) -> None:
    """
    将多个 MRC 文件转换为 HDF5 格式
    
    Args:
        mrc_files: MRC 文件路径列表
        output_path: 输出 HDF5 文件路径
    """
    print(f"开始转换 {len(mrc_files)} 个 MRC 文件到 HDF5...")
    
    with h5py.File(output_path, 'w') as f:
        data_group = f.create_group('data')
        metadata = []
        
        for i, mrc_file in enumerate(mrc_files):
            try:
                with mrcfile.open(mrc_file) as mrc:
                    data = mrc.data.astype(np.float32)
                    
                    # 存储数据
                    data_group.create_dataset(f'volume_{i:06d}', data=data)
                    
                    # 收集元数据
                    metadata.append({
                        'file_path': str(mrc_file),
                        'shape': list(data.shape),
                        'dtype': str(data.dtype),
                        'voxel_size': list(mrc.voxel_size) if mrc.voxel_size else None
                    })
                    
            except Exception as e:
                print(f"警告: 无法处理文件 {mrc_file}: {e}")
                continue
            
            if (i + 1) % 1000 == 0:
                print(f"已处理 {i + 1} 个文件")
        
        # 保存元数据
        f.attrs['num_volumes'] = len(metadata)
        f.attrs['metadata'] = json.dumps(metadata)
    
    print(f"转换完成，保存到: {output_path}")
    print(f"成功处理 {len(metadata)} 个文件")


class HDF5VolumeDataset(Dataset):
    """
    基于 HDF5 文件的 PyTorch Dataset
    
    支持不同尺寸的 3D 体积数据，具有缓存功能以提高访问效率。
    """
    
    def __init__(self, hdf5_path: str, transform: Optional[Callable] = None, cache_size: int = 100):
        """
        初始化 HDF5 Volume Dataset
        
        Args:
            hdf5_path: HDF5 文件路径
            transform: 数据变换函数
            cache_size: 缓存大小（最近访问的数据量）
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size
        
        # 验证文件存在
        if not Path(hdf5_path).exists():
            raise FileNotFoundError(f"HDF5 文件不存在: {hdf5_path}")
        
        # 获取文件信息
        with h5py.File(hdf5_path, 'r') as f:
            self.num_volumes = f.attrs['num_volumes']
            self.data_group = f['data']
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return self.num_volumes
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取指定索引的数据
        
        Args:
            idx: 数据索引
            
        Returns:
            PyTorch tensor 格式的 3D 体积数据
        """
        if idx < 0 or idx >= self.num_volumes:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.num_volumes})")
        
        # 检查缓存
        if idx in self.cache:
            data = self.cache[idx]
        else:
            # 从 HDF5 加载数据
            with h5py.File(self.hdf5_path, 'r') as f:
                data = f['data'][f'volume_{idx:06d}'][:]
            
            # 转换为 PyTorch tensor
            data = torch.from_numpy(data.astype(np.float32))
            
            # 应用变换
            if self.transform:
                data = self.transform(data)
            
            # 缓存管理
            if len(self.cache) >= self.cache_size:
                # 移除最旧的缓存项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[idx] = data
        
        return data
    
    def get_metadata(self) -> List[dict]:
        """
        获取所有数据的元数据
        
        Returns:
            元数据列表
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            return json.loads(f.attrs['metadata'])
    
    def get_volume_info(self, idx: int) -> dict:
        """
        获取指定体积的元数据
        
        Args:
            idx: 体积索引
            
        Returns:
            体积元数据字典
        """
        metadata = self.get_metadata()
        if idx < 0 or idx >= len(metadata):
            raise IndexError(f"索引 {idx} 超出范围 [0, {len(metadata)})")
        return metadata[idx]
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()


def create_sample_mrc_files(output_dir: str, num_files: int = 5) -> List[str]:
    """
    创建示例 MRC 文件用于测试
    
    Args:
        output_dir: 输出目录
        num_files: 创建文件数量
        
    Returns:
        创建的文件路径列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    mrc_files = []
    shapes = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
    
    for i in range(num_files):
        # 随机选择尺寸
        shape = shapes[i % len(shapes)]
        
        # 生成随机数据
        data = np.random.rand(*shape).astype(np.float32)
        
        # 保存为 MRC 文件
        mrc_file = output_dir / f'sample_{i:03d}.mrc'
        with mrcfile.new(str(mrc_file)) as mrc:
            mrc.set_data(data)
            mrc.update_header_from_data()
        
        mrc_files.append(str(mrc_file))
    
    return mrc_files


if __name__ == "__main__":
    # 示例用法
    print("创建示例 MRC 文件...")
    mrc_files = create_sample_mrc_files("test_data", 10)
    
    print("转换为 HDF5...")
    convert_mrcs_to_hdf5(mrc_files, "test_volumes.h5")
    
    print("创建 PyTorch Dataset...")
    dataset = HDF5VolumeDataset("test_volumes.h5", cache_size=5)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"第一个样本形状: {dataset[0].shape}")
    print(f"第二个样本形状: {dataset[1].shape}")
    
    # 清理测试文件
    import shutil
    shutil.rmtree("test_data")
    Path("test_volumes.h5").unlink()
    print("测试完成，清理临时文件")
