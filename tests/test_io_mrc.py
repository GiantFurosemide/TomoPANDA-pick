"""
测试 io_mrc 模块的功能
"""

import unittest
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.io_mrc import convert_mrcs_to_hdf5, HDF5VolumeDataset, create_sample_mrc_files


class TestIOMRC(unittest.TestCase):
    """测试 io_mrc 模块"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.mrc_files = []
        self.hdf5_path = os.path.join(self.temp_dir, 'test_volumes.h5')
    
    def tearDown(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_sample_mrc_files(self):
        """测试创建示例 MRC 文件"""
        mrc_files = create_sample_mrc_files(self.temp_dir, 5)
        
        # 检查文件是否创建
        self.assertEqual(len(mrc_files), 5)
        for mrc_file in mrc_files:
            self.assertTrue(Path(mrc_file).exists())
        
        # 检查文件内容
        import mrcfile
        with mrcfile.open(mrc_files[0]) as mrc:
            self.assertIsInstance(mrc.data, np.ndarray)
            self.assertEqual(len(mrc.data.shape), 3)
    
    def test_convert_mrcs_to_hdf5(self):
        """测试 MRC 文件转换为 HDF5"""
        # 创建示例文件
        mrc_files = create_sample_mrc_files(self.temp_dir, 3)
        
        # 转换为 HDF5
        convert_mrcs_to_hdf5(mrc_files, self.hdf5_path)
        
        # 检查 HDF5 文件
        self.assertTrue(Path(self.hdf5_path).exists())
        
        import h5py
        with h5py.File(self.hdf5_path, 'r') as f:
            # 检查数据结构
            self.assertIn('data', f)
            self.assertIn('num_volumes', f.attrs)
            self.assertEqual(f.attrs['num_volumes'], 3)
            
            # 检查数据
            for i in range(3):
                volume_key = f'volume_{i:06d}'
                self.assertIn(volume_key, f['data'])
                data = f['data'][volume_key][:]
                self.assertIsInstance(data, np.ndarray)
                self.assertEqual(len(data.shape), 3)
    
    def test_hdf5_volume_dataset(self):
        """测试 HDF5VolumeDataset"""
        # 创建示例文件并转换为 HDF5
        mrc_files = create_sample_mrc_files(self.temp_dir, 5)
        convert_mrcs_to_hdf5(mrc_files, self.hdf5_path)
        
        # 创建 Dataset
        dataset = HDF5VolumeDataset(self.hdf5_path, cache_size=3)
        
        # 测试基本功能
        self.assertEqual(len(dataset), 5)
        
        # 测试数据访问
        data = dataset[0]
        self.assertIsInstance(data, torch.Tensor)
        self.assertEqual(len(data.shape), 3)
        
        # 测试缓存
        data1 = dataset[0]
        data2 = dataset[0]
        self.assertTrue(torch.equal(data1, data2))
        
        # 测试元数据
        metadata = dataset.get_metadata()
        self.assertEqual(len(metadata), 5)
        self.assertIn('file_path', metadata[0])
        self.assertIn('shape', metadata[0])
    
    def test_hdf5_volume_dataset_with_transform(self):
        """测试带变换的 HDF5VolumeDataset"""
        # 创建示例文件并转换为 HDF5
        mrc_files = create_sample_mrc_files(self.temp_dir, 3)
        convert_mrcs_to_hdf5(mrc_files, self.hdf5_path)
        
        # 定义变换函数
        def normalize_transform(data):
            return (data - data.mean()) / data.std()
        
        # 创建带变换的 Dataset
        dataset = HDF5VolumeDataset(self.hdf5_path, transform=normalize_transform)
        
        # 测试变换
        data = dataset[0]
        self.assertIsInstance(data, torch.Tensor)
        # 检查是否进行了归一化（均值为0，标准差为1）
        self.assertAlmostEqual(data.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(data.std().item(), 1.0, places=5)
    
    def test_hdf5_volume_dataset_cache(self):
        """测试 Dataset 缓存功能"""
        # 创建示例文件并转换为 HDF5
        mrc_files = create_sample_mrc_files(self.temp_dir, 5)
        convert_mrcs_to_hdf5(mrc_files, self.hdf5_path)
        
        # 创建小缓存的 Dataset
        dataset = HDF5VolumeDataset(self.hdf5_path, cache_size=2)
        
        # 访问数据填充缓存
        data1 = dataset[0]
        data2 = dataset[1]
        data3 = dataset[2]
        
        # 缓存应该只保留最近访问的2个
        self.assertEqual(len(dataset.cache), 2)
        self.assertIn(1, dataset.cache)  # 最近访问的
        self.assertIn(2, dataset.cache)  # 最近访问的
        self.assertNotIn(0, dataset.cache)  # 应该被移除
        
        # 测试清空缓存
        dataset.clear_cache()
        self.assertEqual(len(dataset.cache), 0)
    
    def test_hdf5_volume_dataset_index_error(self):
        """测试 Dataset 索引错误处理"""
        # 创建示例文件并转换为 HDF5
        mrc_files = create_sample_mrc_files(self.temp_dir, 3)
        convert_mrcs_to_hdf5(mrc_files, self.hdf5_path)
        
        dataset = HDF5VolumeDataset(self.hdf5_path)
        
        # 测试负索引
        with self.assertRaises(IndexError):
            dataset[-1]
        
        # 测试超出范围的索引
        with self.assertRaises(IndexError):
            dataset[10]
    
    def test_different_volume_sizes(self):
        """测试不同尺寸的体积数据"""
        # 创建不同尺寸的示例文件
        mrc_files = create_sample_mrc_files(self.temp_dir, 6)  # 会创建不同尺寸
        
        # 转换为 HDF5
        convert_mrcs_to_hdf5(mrc_files, self.hdf5_path)
        
        # 创建 Dataset
        dataset = HDF5VolumeDataset(self.hdf5_path)
        
        # 检查不同尺寸的数据
        shapes = set()
        for i in range(len(dataset)):
            data = dataset[i]
            shapes.add(data.shape)
        
        # 应该有不同的尺寸
        self.assertGreater(len(shapes), 1)
    
    def test_pytorch_dataloader_integration(self):
        """测试与 PyTorch DataLoader 的集成"""
        # 创建相同尺寸的示例文件（避免批处理问题）
        import mrcfile
        mrc_files = []
        for i in range(6):  # 创建6个相同尺寸的文件
            mrc_file = os.path.join(self.temp_dir, f'same_size_{i:03d}.mrc')
            data = np.random.rand(64, 64, 64).astype(np.float32)
            with mrcfile.new(mrc_file) as mrc:
                mrc.set_data(data)
                mrc.update_header_from_data()
            mrc_files.append(mrc_file)
        
        convert_mrcs_to_hdf5(mrc_files, self.hdf5_path)
        
        # 创建 Dataset
        dataset = HDF5VolumeDataset(self.hdf5_path, cache_size=5)
        
        # 创建 DataLoader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=3, 
            shuffle=True,
            num_workers=0  # 避免多进程问题
        )
        
        # 测试 DataLoader
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            self.assertIsInstance(batch, torch.Tensor)
            self.assertEqual(len(batch.shape), 4)  # [batch_size, z, y, x]
            self.assertEqual(batch.shape[0], 3)  # batch_size
            self.assertEqual(batch.shape[1:], (64, 64, 64))  # 所有数据都是相同尺寸
        
        # 应该有一些批次
        self.assertGreater(batch_count, 0)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
