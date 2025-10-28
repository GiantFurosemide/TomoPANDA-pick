# Dynamo 表格（.tbl）格式说明 / Dynamo Table Format

> 参考官方文档（Reference）：[Dynamo Table](https://www.dynamo-em.org/w/index.php?title=Table)

## 简介 / Overview

- Dynamo Table 是 Dynamo 的基础元数据结构，本质上是一个无表头的数值矩阵：每一行代表一个粒子（particle），每一列是预定义属性。
- 第 1 列为粒子标识（tag），与数据文件夹中的粒子文件名相对应，例如 `particle_00003.em` 对应 `tag = 3`。
- 表格在对齐（alignment）、分类与体裁剪（cropping from volumes）等工作流中被广泛使用。

## 文件格式 / File Formats

- 文本表（ASCII）：扩展名 `.tbl`，可直接以文本查看与编辑。
- 二进制表（Binary）：扩展名 `.tblb`（或文档中亦描述为二进制 `.tbl`），用于大型表以显著加速 I/O。

提示：两种格式均可通过 Dynamo 的 `dread` / `dwrite` 读写，二进制仅在超大规模表时推荐。

## 列约定 / Column Convention

行表示粒子；列为固定意义的属性。常用列如下（单位见说明；可在 MATLAB 中用 `dthelp` 查询完整定义）：

- 1: tag 粒子标识（对应数据文件名中的编号）
- 2: aligned 是否参与对齐（1 表示参与）
- 3: averaged 是否被用于平均
- 4-6: dx, dy, dz 相对中心的像素位移（pixels）
- 7-9: tdrot, tilt, narot 欧拉角（degrees），顺序为 Z-X-Z（Dynamo 约定）
- 10: cc 互相关系数（cross-correlation）
- 11: cc2 阈值化后的互相关系数 II
- 12: cpu 执行对齐的处理器编号
- 13: ftype 频率采样/倾转模式标志（详见 Dynamo 帮助）
- 14-17: ymintilt, ymaxtilt, xmintilt, xmaxtilt 倾转角范围（degrees）
- 18-19: fs1, fs2 傅里叶采样自由参数
- 20: tomo 所属断层编号（tomogram number）
- 21: reg 断层内区域标号（region）
- 22: class 分类编号
- 23: annotation 任意自定义标签
- 24-26: x, y, z 原始体数据坐标（体素坐标）
- 27: dshift 位移向量范数
- 28: daxis 轴向差异
- 29: dnarot narot 差值
- 30: dcc cc 差值
- 31: otag 原始标识（子框选 subboxing）
- 32: npar 合并粒子数/子单元标签（subboxing）
- 34: ref 参考编号（多参考项目）
- 35: sref 子参考（如 PCA 生成）
- 36: apix 每像素埃（Å/px）
- 37: def 离焦（micron）
- 41-42: eig1, eig2 特征系数（eigencoefficients）

注：用户可增加额外列作为自定义属性。列意义固定但可拓展，使用时请与下游工具保持一致。

## 最小示例 / Minimal Example

以下为 3 行示例（ASCII `.tbl`，无表头；为演示清晰，采用空格分隔；实际文件不应包含注释）：

```text
3  1 1  0 0 0   0 0 0   0 0   0  -60 60  -60 60   0 0   1  0 0   120.0 240.0 360.0   0 0 0 0   0 0   1   0   1.35   2.0   0 0
4  1 1  2 1 0  10 5 0   0 0   0  -60 60  -60 60   0 0   1  0 0   220.0 140.0 260.0   0 0 0 0   0 0   1   0   1.35   2.0   0 0
5  0 1 -1 0 3  20 0 5   0 0   0  -60 60  -60 60   0 0   2  0 0    50.0  80.0  90.0   0 0 0 0   0 0   1   0   1.35   2.0   0 0
```

- 第 1 列为 tag，与数据文件夹中的 `particle_00003.em` 等对应。
- 列 24-26 为体素坐标 x, y, z；列 7-9 为欧拉角 tdrot, tilt, narot（度）。

## 常见操作 / Common Operations

### MATLAB 读写 / MATLAB IO

```matlab
% 读取（支持 .tbl 与 .tblb）
T = dread('mytable.tbl');

% 写入 ASCII 表
dwrite(T, 'output.tbl');

% 写入二进制表（适合大型数据）
dwrite(T, 'output.tblb');
```

### 基本信息统计 / Quick Stats

```matlab
dtinfo('mytable.tbl');  % 输出各列均值/方差等统计
```

### 可视化 / Visualization

```matlab
% 命令行绘图（示例：带方向的空间分布）
dtplot('example.tbl', '-pf', 'oriented_positions');

% 轻量 GUI
dtshow('file.tbl');

% 复杂浏览器（二维属性联合可视化）
dtview('file.tbl');
```

### 从体数据裁剪粒子 / Cropping from Volumes

- 任意表只要包含列 24-26（x, y, z）即可用于从体或多体中裁剪粒子。
- 多体联合使用需配合体-表映射文件（volume-table mapping）。

### 选择与合并 / Selection and Merging

```matlab
% 选择：使用 dtgrep 可按 tag 范围或方向等条件筛选
dtgrep('all.tbl', 'tag', [1 1000]);

% 低级合并：确保 tag 不重复
t1 = dread('table1.tbl');
t2 = dread('table2.tbl');
tMerged = cat(1, t1, t2);
dwrite(tMerged, 'merged.tbl');

% 检查是否有重复 tag
length(unique(tMerged(:,1)))

% 高级合并命令
dynamo_table_merge('table1.tbl', 'table2.tbl', 'merged.tbl');
```

### 创建空表 / Create Blank Table

```matlab
% 基于数据文件夹创建兼容空表（含正确列数与默认值）
Tblank = dynamo_table_blank('myDataFolder');
```

## 约定与单位 / Conventions and Units

- 角度：tdrot, tilt, narot 为度（degrees），欧拉顺序 Z-X-Z。
- 位移：dx, dy, dz 为像素（pixels）。
- 坐标：x, y, z 与原始体素坐标系一致（通常从 0 或 1 起始，具体视下游工具而定）。
- 标识：tag 必须与数据文件夹中的粒子文件一致，合并表时禁止重复。

## 与工作流的关系 / Workflow Notes

- 对齐/分类：aligned、averaged、cc/cc2、角度与位移列由对齐迭代更新。
- 多参考：多参考项目使用 ref（列 34）标识参考。
- 子框选：otag（列 31）与 npar（列 32）在子框选/子单元场景中使用。

## 常见陷阱 / Pitfalls

- 忘记填充列 24-26（坐标）会导致体裁剪失败。
- 合并表后未检查 tag 唯一性。
- 欧拉角顺序与单位不一致导致朝向解释错误。
- 极大表使用 ASCII 导致 I/O 过慢，建议 `.tblb`。

## 参考 / References

- 官方说明（Official doc）：[Dynamo Table](https://www.dynamo-em.org/w/index.php?title=Table)
- 更多命令：在 MATLAB/Octave 中查看 `dthelp`, `dapropos table`。
