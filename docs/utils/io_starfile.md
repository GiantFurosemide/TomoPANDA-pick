# STAR 文件操作工具函数

基于 `starfile` 库的常用 STAR 文件操作函数列表。

## 文件读取和写入

### `starfile.read(filename, read_n_blocks=None, always_dict=False, parse_as_string=[])`
- **输入**: `filename: PathLike`, `read_n_blocks: Optional[int]`, `always_dict: bool`, `parse_as_string: List[str]`
- **输出**: `Union[DataBlock, Dict[DataBlock]]`
- **说明**: 从 STAR 文件读取数据。基本数据块读取为字典，循环块读取为 pandas DataFrame。多个数据块时返回字典，单个数据块时默认返回块本身

### `starfile.write(data, filename, float_format='%.6f', sep='\t', na_rep='<NA>', quote_character='"', quote_all_strings=False)`
- **输入**: `data: Union[DataBlock, Dict[str, DataBlock], List[DataBlock]]`, `filename: PathLike`, `float_format: str`, `sep: str`, `na_rep: str`, `quote_character: str`, `quote_all_strings: bool`
- **输出**: `None`
- **说明**: 将数据写入 STAR 格式文件。支持字典、DataFrame 或数据块列表

### `starfile.to_string(data, float_format='%.6f', sep='\t', na_rep='<NA>', quote_character='"', quote_all_strings=False)`
- **输入**: `data: Union[DataBlock, Dict[str, DataBlock], List[DataBlock]]`, `float_format: str`, `sep: str`, `na_rep: str`, `quote_character: str`, `quote_all_strings: bool`
- **输出**: `str`
- **说明**: 将数据转换为 STAR 格式字符串，不写入文件

## 数据块类型

### 基本数据块 (Basic Data Block)
- **类型**: `Dict[str, Any]`
- **说明**: 包含键值对的基本数据块，通常存储元数据信息
- **示例**: `{'rlnImageSizeX': 64, 'rlnImageSizeY': 64, 'rlnImageSizeZ': 64}`

### 循环数据块 (Loop Data Block)
- **类型**: `pandas.DataFrame`
- **说明**: 包含表格数据的循环块，每行代表一个条目，列对应不同的标签
- **示例**: 粒子坐标、角度、微图名称等

## 数据访问和操作

### 读取单个数据块
```python
# 读取单个数据块（默认行为）
df = starfile.read('particles.star')
print(df.head())
```

### 读取多个数据块
```python
# 强制返回字典格式，即使只有一个数据块
data = starfile.read('particles.star', always_dict=True)
print(data.keys())  # 显示所有数据块名称
```

### 限制读取数据块数量
```python
# 只读取前 n 个数据块
data = starfile.read('large_file.star', read_n_blocks=5)
```

### 字符串解析控制
```python
# 指定某些列保持字符串格式，不转换为数值
data = starfile.read('particles.star', parse_as_string=['rlnMicrographName', 'rlnImageName'])
```

## 数据写入操作

### 写入单个数据块
```python
import pandas as pd

# 创建 DataFrame
df = pd.DataFrame({
    'rlnCoordinateX': [91.8, 97.6, 92.4],
    'rlnCoordinateY': [83.6, 80.4, 88.8],
    'rlnCoordinateZ': [203.3, 203.1, 210.7],
    'rlnMicrographName': ['01_10.00Apx.mrc', '01_10.00Apx.mrc', '01_10.00Apx.mrc']
})

# 写入文件
starfile.write(df, 'output_particles.star')
```

### 写入多个数据块
```python
# 创建包含多个数据块的字典
data_blocks = {
    'particles': particles_df,
    'micrographs': micrographs_df,
    'optics': optics_dict
}

# 写入文件
starfile.write(data_blocks, 'multi_block.star')
```

### 自定义格式参数
```python
# 自定义浮点数格式和分隔符
starfile.write(df, 'output.star', 
               float_format='%.3f',  # 3位小数
               sep=' ',              # 空格分隔
               na_rep='NULL')        # 空值表示
```

## 数据操作示例

### 修改粒子坐标
```python
# 读取粒子数据
df = starfile.read('particles.star')

# 修改坐标
df['rlnCoordinateX'] += 10  # X坐标偏移10像素
df['rlnCoordinateY'] += 5   # Y坐标偏移5像素

# 保存修改后的数据
starfile.write(df, 'modified_particles.star')
```

### 筛选特定微图的粒子
```python
# 筛选特定微图的粒子
filtered_df = df[df['rlnMicrographName'] == '01_10.00Apx.mrc']

# 保存筛选结果
starfile.write(filtered_df, 'filtered_particles.star')
```

### 添加新列
```python
# 添加新的标签列
df['rlnClassNumber'] = 1
df['rlnDefocusU'] = 2000.0
df['rlnDefocusV'] = 2000.0

# 保存更新后的数据
starfile.write(df, 'updated_particles.star')
```

## 高级用法

### 处理大型文件
```python
# 分批读取大型文件
for i in range(0, total_blocks, batch_size):
    data = starfile.read('large_file.star', read_n_blocks=batch_size)
    # 处理数据块
    process_data_block(data)
```

### 合并多个 STAR 文件
```python
import pandas as pd

# 读取多个文件
files = ['file1.star', 'file2.star', 'file3.star']
all_data = []

for file in files:
    df = starfile.read(file)
    all_data.append(df)

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)

# 保存合并结果
starfile.write(combined_df, 'combined_particles.star')
```

### 数据验证和检查
```python
# 检查必需列是否存在
required_columns = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"缺少必需列: {missing_columns}")
else:
    print("所有必需列都存在")

# 检查数据范围
print(f"X坐标范围: {df['rlnCoordinateX'].min()} - {df['rlnCoordinateX'].max()}")
print(f"Y坐标范围: {df['rlnCoordinateY'].min()} - {df['rlnCoordinateY'].max()}")
```

## 与 RELION 的兼容性

### 标准 RELION 标签
```python
# 常用的 RELION 标签
relion_labels = {
    'coordinates': ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'],
    'angles': ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'],
    'micrograph': ['rlnMicrographName'],
    'image': ['rlnImageName'],
    'class': ['rlnClassNumber'],
    'defocus': ['rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle'],
    'ctf': ['rlnCtfFigureOfMerit', 'rlnCtfMaxResolution'],
    'particle': ['rlnParticleName', 'rlnGroupNumber']
}
```

### 创建 RELION 兼容文件
```python
# 创建符合 RELION 格式的粒子文件
def create_relion_particles(coordinates, micrograph_name, output_file):
    df = pd.DataFrame({
        'rlnCoordinateX': coordinates[:, 0],
        'rlnCoordinateY': coordinates[:, 1], 
        'rlnCoordinateZ': coordinates[:, 2],
        'rlnAngleRot': 0.0,
        'rlnAngleTilt': 0.0,
        'rlnAnglePsi': 0.0,
        'rlnMicrographName': micrograph_name,
        'rlnClassNumber': 1,
        'rlnDefocusU': 2000.0,
        'rlnDefocusV': 2000.0,
        'rlnDefocusAngle': 0.0
    })
    
    starfile.write(df, output_file)
```

## 使用示例

```python
import starfile
import pandas as pd
import numpy as np

# 读取 STAR 文件
df = starfile.read('particles.star')
print(f"粒子数量: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 数据操作
df['rlnCoordinateX'] += 10  # 坐标偏移
df['rlnClassNumber'] = 1    # 添加分类标签

# 保存修改后的文件
starfile.write(df, 'modified_particles.star')

# 创建新的 STAR 文件
new_data = {
    'rlnImageSizeX': 64,
    'rlnImageSizeY': 64,
    'rlnImageSizeZ': 64,
    'rlnPixelSize': 1.0
}

starfile.write(new_data, 'optics.star')

# 转换为字符串格式
star_string = starfile.to_string(df)
print(star_string[:500])  # 显示前500个字符
```

## 注意事项

1. STAR 文件格式主要用于 RELION 软件，确保标签名称符合 RELION 标准
2. 循环数据块会自动转换为 pandas DataFrame，便于数据操作
3. 基本数据块保持为字典格式，适合存储元数据
4. 大文件建议分批处理，避免内存溢出
5. 字符串列使用 `parse_as_string` 参数防止自动类型转换
6. 浮点数精度可通过 `float_format` 参数控制
7. 空值处理可通过 `na_rep` 参数自定义
8. 多数据块文件建议使用 `always_dict=True` 确保返回字典格式
