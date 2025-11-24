# 基于球面 Fibonacci 均匀方向采样的 subtomo 多取向投影生成文档

**使用：`python script.py -i config.yaml`**

**必生成：RELION particle.star（ZYZ 欧拉角）**

---

# 0. 任务概述

输入：

* 一个 `.txt` 文件，每行是一个 subtomo 的 `.mrc` 路径
* Fibonacci 均匀采样数量 N
* 输出 root 路径

输出：

* 每个取向一个 `.mrcs` 投影栈
* 每个取向一个 local index
* 一个 global index
* 一个 orientations.tsv
* 一个 **RELION-compatible particle.star（ZYZ 欧拉角）**

---

# 1. YAML 配置格式（更新版）

入口：

```bash
python script.py -i config.yaml
```

配置：

```yaml
input:
  subtomo_txt: /path/to/subtomos.txt   # txt 文件，每行一个 MRC 绝对或相对路径

orientations:
  N_orientations: 10
  psi_deg: 0.0                         # ZYZ 第三角 Psi（固定 in-plane）

output:
  root_dir: /path/to/output_root
  orientation_dir_pattern: "ori_{k:03d}"
  stack_name_pattern: "proj_ori_{k:03d}.mrcs"
  orientations_table: "orientations.tsv"
  global_index_table: "index_particles.tsv"
  per_orientation_index_pattern: "index_ori_{k:03d}.tsv"
  relion_star: "particles.star"        # 必须生成，不可为空

projection:
  box_size: 64
  mode: "sum"                          # sum | mean | max | central_slice
  normalize: true

optics:
  pixel_size: 3.5
  voltage_kv: 300
  cs_mm: 2.7
  amplitude_contrast: 0.1
  optics_group_name: "opticsGroup1"
  optics_group_id: 1
```

---

# 2. 脚本内部阶段

按以下顺序执行：

1. 读取 YAML
2. **读取 subtomo_txt（每行一个路径）** → 生成 SubtomoTable
3. 生成 N 个 Fibonacci 球面方向
4. 方向向量 → ZYZ 欧拉角（RELION）
5. 构造全局粒子计划
6. 按方向生成 `.mrcs` 栈
7. 写 per-orientation index
8. 写 global index
9. **写 particle.star（强制执行）**

---

# 3. 读取 subtomo 列表（由 txt 文件提供）

TXT 示例内容：

```
/data/tomo/subtomo_0001.mrc
/data/tomo/subtomo_0002.mrc
...
```

解析逻辑：

* 一行 = 一个 `.mrc` 文件路径
* 自动 strip 换行符
* 去除空行
* 生成表：
  * `subtomo_id`（0-based or 1-based）
  * `subtomo_path`（绝对或相对 path）

---

# 4. Fibonacci 球面采样（均匀方向生成）

对 `k = 0 ... N-1`：

[

t_k=\frac{k+0.5}{N},\quad z_k=1-2t_k

]

[

r_k=\sqrt{1 - z_k^2}

]

[

\varphi=\frac{1+\sqrt5}{2},\quad \phi_k = 2\pi\frac{k}{\varphi}

]

[

x_k=r_k\cos\phi_k,\quad y_k=r_k\sin\phi_k

]

方向向量 ((vx,vy,vz))。

---

# 5. 方向向量 → RELION ZYZ 欧拉角（严格遵守 RELION convention）

RELION ZYZ：

[

R = R_Z(\mathrm{Psi}) R_Y(\mathrm{Tilt}) R_Z(\mathrm{Rot})

]

目标：

旋转后把 (Z) 轴对准方向 (v_k)。

转换公式：

1）Tilt（绕 Y）

[

\mathrm{Tilt}_k = \arccos(vz)

]

2）Rot（绕 Z）

[

\mathrm{Rot}_k = \mathrm{atan2}(vy, vx)

]

3）Psi（绕新 Z）

[

\mathrm{Psi}_k = \psi_0  \quad (\text{配置给定，通常=0})

]

4）全部转 degree。

写入 `orientations.tsv`：

```
orientation_id  vx vy vz  rot_deg  tilt_deg  psi_deg
0               ...      ...       0.0
1               ...      ...       0.0
...
```

---

# 6. 全局粒子计划

给定：

* subtomo 数 M
* 取向数 N

生成 M×N 条记录。

字段：

* `particle_global_id`
* `subtomo_id`
* `subtomo_path`
* `orientation_id`
* `rot_deg tilt_deg psi_deg`
* `vx vy vz`
* `stack_relpath = ori_{k}/proj_ori_{k}.mrcs`
* `slice_index_in_stack = subtomo_id 的顺序（1-based）`

---

# 7. 生成投影栈（按取向）

对每个 k：

1. 创建目录：`ori_{k}/`
2. 创建新的 `.mrcs` 文件 `proj_ori_{k}.mrcs`
3. 对每个 subtomo：
   * 读取 mrc 3D 体
   * 构造 ZYZ 旋转矩阵
   * 3D → 2D 投影：sum / mean / max / central_slice
   * 重采样至 box_size×box_size
   * 写入 `.mrcs` 的 slice = subtomo_id

---

# 8. per-orientation index：`index_ori_{k}.tsv`

字段：

```
slice_index  subtomo_id  subtomo_path  orientation_id  rot_deg  tilt_deg  psi_deg
```

---

# 9. 全局 index：`index_particles.tsv`

字段：

```
particle_global_id
orientation_id
subtomo_id
subtomo_path
stack_relpath
slice_index_in_stack
rot_deg tilt_deg psi_deg
vx vy vz
```

---

# 10. 必须生成 RELION 3 particle.star（强制）

分两块：

## 10.1 optics block

```
data_optics

loop_
_rlnOpticsGroupName
_rlnOpticsGroup
_rlnImagePixelSize
_rlnVoltage
_rlnSphericalAberration
_rlnAmplitudeContrast
_rlnImageSize
opticsGroup1  1  3.5  300  2.7  0.1 48
```

字段来自 YAML。

---

## 10.2 particles block（从 global index 构造）

```
data_particles

loop_
_rlnImageName
_rlnAngleRot
_rlnAngleTilt
_rlnAnglePsi
_rlnOpticsGroup
```

每条粒子记录一行：

```
{slice_index_in_stack}@{stack_relpath}   rot   tilt   psi   optics_group_id
```

例如：

```
1@ori_000/proj_ori_000.mrcs   12.34   57.89   0.0   1
2@ori_000/proj_ori_000.mrcs   12.34   78.12   0.0   1
...
1@ori_001/proj_ori_001.mrcs   ...
```

要求：

* 角度必须是 **degree**
* ZYZ 必须严格遵守
* `_rlnImageName` 路径必须是相对于 star 所在目录的相对路径
* cryoSPARC 可直接 import

---

# 11. 最终输出结构（更新版）

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
  ori_{N-1}/
    proj_ori_{N-1}.mrcs
    index_ori_{N-1}.tsv
```

---



# 12. 使用方法

```bash
conda activate tomopanda
python utils/2d_projection/main.py -i config.yaml
```


# 13. 可选验证步骤

1. 检查头三个方向，如 (0,0,1),(1,0,0),(0,1,0)
2. 在 RELION 中：
   ```
   relion_display --i particles.star
   ```
3. 在 cryoSPARC 中：Import particles (RELION 3 STAR)
4. 在每个 ori_k 中，用 `relion_display --i proj_ori_k.mrcs` 验证 slice 顺序

---

# 14. 结果分析工具（analyze_results.py）

`analyze_results.py` 提供了从 particle star 文件中提取 slice index，以及从 txt 文件中提取指定行的功能。

## 14.1 主要功能

1. **从 star 文件提取 slice indices**
   - 从 RELION particle star 文件中提取所有的 slice index（n 值）
   - star 文件中的 `_rlnImageName` 格式为：`n@x.mrcs`，其中 n 是 slice index（从 1 开始）
   - 可以将 0-based 的 indices 保存到 txt 文件（每行一个 index，从 0 开始）

2. **从 txt 文件提取指定行**
   - 支持 0-based 或 1-based 的 indices
   - 可以保存提取的行到新的 txt 文件

3. **整合功能**
   - 从 star 文件提取 indices，然后从 txt 文件提取对应的行
   - 输入：star 文件和 txt 文件
   - 输出：提取的行保存到 txt 文件
   - 可选：同时保存 0-based indices 到单独的 txt 文件

## 14.2 Python API 使用

```python
from utils.2d_projection.analyze_results import (
    extract_slice_indices_from_star,
    extract_lines_by_indices,
    extract_lines_from_star_and_txt
)

# 方法1：只提取indices并保存到文件（0-based）
indices = extract_slice_indices_from_star("particles.star", "indices.txt")
# indices.txt 内容：每行一个0-based的index

# 方法2：从txt文件提取指定行（使用1-based indices，与star文件对应）
lines = extract_lines_by_indices("subtomos.txt", [1, 3, 5], "output.txt")

# 方法3：从txt文件提取指定行（使用0-based indices）
lines = extract_lines_by_indices("subtomos.txt", [0, 2, 4], "output.txt", indices_are_0based=True)

# 方法4：整合功能 - 从star提取indices，然后从txt提取对应行
lines = extract_lines_from_star_and_txt(
    "particles.star",
    "subtomos.txt", 
    "extracted_lines.txt",
    "indices.txt"  # 可选：保存0-based indices
)
```

## 14.3 命令行使用

```bash
# 从star文件提取indices并保存（0-based）
python utils/2d_projection/analyze_results.py -s particles.star --index-file indices.txt

# 整合功能：从star和txt提取行
python utils/2d_projection/analyze_results.py -s particles.star -t subtomos.txt -o output.txt --index-file indices.txt

# 从txt文件提取指定行（使用1-based indices）
python utils/2d_projection/analyze_results.py -t subtomos.txt -i 1 3 5 -o output.txt

# 从txt文件提取指定行（使用0-based indices）
python utils/2d_projection/analyze_results.py -t subtomos.txt -i 0 2 4 -o output.txt --zero-based

# 从txt文件提取颗粒ID，然后从tbl文件提取对应的行
python utils/2d_projection/analyze_results.py --extract-tbl-by-txt -t particles.txt --tbl all_particles.tbl --output-tbl filtered_particles.tbl
```

## 14.4 从颗粒路径提取ID并过滤tbl文件

```python
from utils.2d_projection.analyze_results import (
    extract_particle_ids_from_txt,
    filter_dynamo_tbl_by_particle_ids,
    extract_tbl_by_particle_txt
)

# 方法1：从txt文件提取颗粒ID
particle_ids = extract_particle_ids_from_txt("particles.txt")
# 颗粒路径格式：particle_035948.mrc -> 提取出 35948（去掉前导零）

# 方法2：从tbl文件提取指定颗粒ID的行
filtered_df = filter_dynamo_tbl_by_particle_ids(
    "all_particles.tbl",
    [35948, 1234, 5678],
    "filtered_particles.tbl"
)

# 方法3：整合功能 - 从txt提取颗粒ID，然后从tbl提取对应行
filtered_df = extract_tbl_by_particle_txt(
    "particles.txt",
    "all_particles.tbl",
    "filtered_particles.tbl"
)
```

## 14.5 注意事项

- star 文件中的 slice index（`n@x.mrcs` 中的 `n`）是从 1 开始的（1-based）
- 保存到 txt 文件的 indices 会自动转换为 0-based（n-1）
- 从 txt 文件读取行时，默认使用 1-based indices（与 star 文件对应）
- 如果使用 `--zero-based` 参数，则使用 0-based indices
- 颗粒路径格式应为：`particle_035948.mrc`（particle_前缀 + 数字ID + 扩展名）
- 颗粒ID会自动去掉前导零（如035948 -> 35948）
- tbl 文件的第一列（tag列）必须是颗粒ID
