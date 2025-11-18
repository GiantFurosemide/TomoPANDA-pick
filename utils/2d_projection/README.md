# 2D Projection Module

基于球面 Fibonacci 均匀方向采样的 subtomo 多取向投影生成模块。

## 文件结构

- `math_utils.py`: 数学工具模块
  - `fibonacci_sphere_sampling()`: Fibonacci 球面均匀方向采样
  - `direction_to_relion_zyz()`: 方向向量到 RELION ZYZ 欧拉角转换
  - `zyz_euler_to_rotation_matrix()`: ZYZ 欧拉角到旋转矩阵转换

- `projection.py`: 3D 到 2D 投影模块
  - `project_3d_to_2d_rotated()`: 完整的 3D 旋转和投影实现

- `main.py`: 主脚本
  - 读取 YAML 配置
  - 生成投影栈
  - 输出 RELION 3 格式的 star 文件

- `example_config.yaml`: 示例配置文件

## 使用方法

```bash
python utils/2d_projection/main.py -i config.yaml
```

## 配置文件格式

参见 `example_config.yaml` 文件。

## 输出结构

```
output_root/
  config_resolved.yaml
  orientations.tsv
  index_particles.tsv
  particles.star
  ori_000/
    proj_ori_000.mrcs
    index_ori_000.tsv
  ori_001/
    proj_ori_001.mrcs
    index_ori_001.tsv
  ...
```

## 依赖

- numpy
- scipy
- pandas
- mrcfile
- starfile
- pyyaml
- tqdm

所有依赖都在项目的 `requirements.txt` 中。

