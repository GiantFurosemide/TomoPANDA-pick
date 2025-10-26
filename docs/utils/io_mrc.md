# MRC/MRCS 文件操作工具函数

基于 `mrcfile` 库的常用 MRC/MRCS 文件操作函数列表。

## 文件打开和创建

### `mrcfile.open(filename, mode='r', permissive=False)`
- **输入**: `filename: str`, `mode: str`, `permissive: bool`
- **输出**: `MrcFile` 对象
- **说明**: 打开现有的 MRC 文件，支持 'r'（只读）、'r+'（读写）、'w+'（新建）模式

### `mrcfile.new(filename, overwrite=False, compression=None)`
- **输入**: `filename: str`, `overwrite: bool`, `compression: str`
- **输出**: `MrcFile` 对象
- **说明**: 创建新的 MRC 文件，支持 gzip 和 bzip2 压缩

### `mrcfile.read(filename)`
- **输入**: `filename: str`
- **输出**: `numpy.ndarray`
- **说明**: 快速读取 MRC 文件数据，返回 numpy 数组

### `mrcfile.write(filename, data, overwrite=False)`
- **输入**: `filename: str`, `data: numpy.ndarray`, `overwrite: bool`
- **输出**: `None`
- **说明**: 快速写入数据到 MRC 文件

## 数据访问和操作

### `mrcfile.open_async(filename, mode='r', permissive=False)`
- **输入**: `filename: str`, `mode: str`, `permissive: bool`
- **输出**: `Future` 对象
- **说明**: 异步打开 MRC 文件，适用于大文件或批量处理

### `mrcfile.mrcfile.MrcMemmap(filename, mode='r')`
- **输入**: `filename: str`, `mode: str`
- **输出**: `MrcMemmap` 对象
- **说明**: 使用内存映射方式打开大文件，支持快速随机访问

## 文件验证

### `mrcfile.validate(filename, print_file=None)`
- **输入**: `filename: str`, `print_file: TextIO`
- **输出**: `bool`
- **说明**: 验证 MRC 文件格式是否正确

## MrcFile 对象属性和方法

### 属性访问

### `mrc.header`
- **类型**: `MrcHeader` 对象
- **说明**: 访问 MRC 文件头部信息

### `mrc.data`
- **类型**: `numpy.ndarray`
- **说明**: 访问 MRC 文件数据数组

### `mrc.extended_header`
- **类型**: `numpy.ndarray`
- **说明**: 访问扩展头部信息

### `mrc.voxel_size`
- **类型**: `tuple`
- **说明**: 获取体素大小 (x, y, z)

### 数据操作方法

### `mrc.set_data(data)`
- **输入**: `data: numpy.ndarray`
- **输出**: `None`
- **说明**: 设置 MRC 文件的数据数组

### `mrc.set_extended_header(extended_header)`
- **输入**: `extended_header: numpy.ndarray`
- **输出**: `None`
- **说明**: 设置扩展头部信息

### `mrc.update_header_from_data()`
- **输入**: `None`
- **输出**: `None`
- **说明**: 根据数据自动更新头部信息

### `mrc.update_header_stats()`
- **输入**: `None`
- **输出**: `None`
- **说明**: 更新头部统计信息（最小值、最大值、平均值等）

### 文件操作方法

### `mrc.flush()`
- **输入**: `None`
- **输出**: `None`
- **说明**: 将数据刷新到磁盘但保持文件打开

### `mrc.close()`
- **输入**: `None`
- **输出**: `None`
- **说明**: 关闭文件并保存所有更改

### 数据类型检查方法

### `mrc.is_single_image()`
- **输入**: `None`
- **输出**: `bool`
- **说明**: 检查是否为单张图像

### `mrc.is_image_stack()`
- **输入**: `None`
- **输出**: `bool`
- **说明**: 检查是否为图像堆栈

### `mrc.is_volume()`
- **输入**: `None`
- **输出**: `bool`
- **说明**: 检查是否为3D体积数据

### `mrc.is_volume_stack()`
- **输入**: `None`
- **输出**: `bool`
- **说明**: 检查是否为3D体积堆栈

### 专用设置方法

### `mrc.set_image_stack(data)`
- **输入**: `data: numpy.ndarray`
- **输出**: `None`
- **说明**: 设置图像堆栈数据并更新相关头部信息

### `mrc.set_volume(data)`
- **输入**: `data: numpy.ndarray`
- **输出**: `None`
- **说明**: 设置3D体积数据并更新相关头部信息

### 调试和验证方法

### `mrc.print_header()`
- **输入**: `None`
- **输出**: `None`
- **说明**: 打印头部信息到控制台

### `mrc.validate()`
- **输入**: `None`
- **输出**: `bool`
- **说明**: 验证当前 MRC 对象是否有效

## 压缩文件支持

### `mrcfile.gzipmrcfile.GzipMrcFile(filename, mode='r')`
- **输入**: `filename: str`, `mode: str`
- **输出**: `GzipMrcFile` 对象
- **说明**: 直接操作 gzip 压缩的 MRC 文件

### `mrcfile.bzip2mrcfile.Bzip2MrcFile(filename, mode='r')`
- **输入**: `filename: str`, `mode: str`
- **输出**: `Bzip2MrcFile` 对象
- **说明**: 直接操作 bzip2 压缩的 MRC 文件

## 使用示例

```python
import mrcfile
import numpy as np

# 读取 MRC 文件
with mrcfile.open('data.mrc') as mrc:
    data = mrc.data
    header = mrc.header
    voxel_size = mrc.voxel_size

# 创建新的 MRC 文件
data = np.random.rand(64, 64, 64).astype(np.float32)
with mrcfile.new('output.mrc') as mrc:
    mrc.set_data(data)
    mrc.update_header_from_data()

# 快速读写
data = mrcfile.read('input.mrc')
mrcfile.write('output.mrc', data)

# 验证文件
is_valid = mrcfile.validate('data.mrc')
```

## PyTorch Dataset + HDF5 方案

### `convert_mrcs_to_hdf5(mrc_files, output_path)`
- **输入**: `mrc_files: List[str]`, `output_path: str`
- **输出**: `None`
- **说明**: 将多个 MRC 文件转换为 HDF5 格式，支持不同尺寸的数据

### `HDF5VolumeDataset(hdf5_path, transform=None, cache_size=100)`
- **输入**: `hdf5_path: str`, `transform: callable`, `cache_size: int`
- **输出**: PyTorch Dataset 对象
- **说明**: 基于 HDF5 文件的 PyTorch Dataset，支持缓存和变换

### 使用示例
```python
from utils.io_mrc import convert_mrcs_to_hdf5, HDF5VolumeDataset
import torch
from torch.utils.data import DataLoader

# 转换 MRC 文件为 HDF5
mrc_files = ['file1.mrc', 'file2.mrc', 'file3.mrc']
convert_mrcs_to_hdf5(mrc_files, 'volumes.h5')

# 创建 PyTorch Dataset
dataset = HDF5VolumeDataset('volumes.h5', cache_size=50)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 训练循环
for batch in dataloader:
    print(f"Batch shape: {batch.shape}")
    # 进行训练...
```

## 注意事项

1. 使用 `with` 语句确保文件正确关闭
2. 大文件建议使用 `MrcMemmap` 进行内存映射
3. 压缩文件会自动检测文件扩展名
4. 修改数据后记得调用 `update_header_from_data()` 或 `update_header_stats()`
5. 异步操作适用于批量处理多个文件
6. HDF5 方案适合大量小文件（20万-50万个），支持不同尺寸数据
