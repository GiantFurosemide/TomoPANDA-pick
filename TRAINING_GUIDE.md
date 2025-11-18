# TomoPANDA-pick 训练指南 / Training Guide

## 1. 安装依赖 / Installation

### 1.0 检查 Python 版本（重要！）
```bash
# 确保使用 Python 3.10 或更高版本（推荐 3.12）
python3 --version  # 应该显示 Python 3.10.x, 3.11.x 或 3.12.x
# 或
python --version   # 如果 python 指向 Python 3

# 如果显示 Python 2.x，请使用 python3 命令
# 推荐使用 Python 3.12 以获得最佳性能和最新特性
```

### 1.1 创建虚拟环境（推荐）
```bash
# 使用 conda（推荐使用 Python 3.12）
conda create -n tomopanda-pick python=3.12
conda activate tomopanda-pick

# 或使用 Python 3.10/3.11（也支持）
conda create -n tomopanda-pick python=3.10
conda activate tomopanda-pick

# 或使用 venv
python3.12 -m venv venv  # 推荐使用 3.12
# 或
python3.10 -m venv venv  # 也支持 3.10
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 验证虚拟环境中的 Python 版本
python --version  # 应该显示 Python 3.10.x, 3.11.x 或 3.12.x
```

### 1.2 安装项目依赖
```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者以开发模式安装项目本身
pip install -e .

# 注意：项目支持 Python 3.10, 3.11, 3.12
# 推荐使用 Python 3.12 以获得最佳性能
```

### 1.3 验证安装
```bash
# 检查 PyTorch 是否安装成功
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 检查 CUDA 是否可用（如果有 GPU）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 2. 准备数据 / Data Preparation

### 2.1 数据格式选择

项目支持两种数据格式：

#### 选项 A：完整 Tomogram + 标注文件（用于粒子检测任务）
- 完整的断层扫描数据
- JSON 格式的标注文件，包含粒子位置

#### 选项 B：已提取的 Subtomogram 颗粒（推荐，用于已提取的颗粒数据）⭐
- 每个文件是一个已提取的颗粒
- 不需要标注文件
- 适合已经完成粒子提取的数据

### 2.2 选项 A：完整 Tomogram 数据格式

**数据目录结构：**
```
data/
├── processed/
│   ├── train/
│   │   ├── tomogram1.mrc
│   │   ├── tomogram1.json
│   │   ├── tomogram2.mrc
│   │   ├── tomogram2.json
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
```

**文件格式要求：**
- **Tomogram 文件**：支持 `.mrc`, `.em`, `.h5`, `.hdf5` 格式
- **Annotation 文件（JSON）**：
```json
{
  "particles": [
    {
      "center": [x, y, z],
      "radius": 5
    }
  ]
}
```

### 2.3 选项 B：已提取的 Subtomogram 颗粒（推荐）⭐

**数据目录结构：**
```
data/
├── processed/
│   ├── train/
│   │   ├── particle_001.mrc
│   │   ├── particle_002.mrc
│   │   ├── particle_003.mrc
│   │   └── ...  # 每个文件是一个颗粒
│   ├── val/
│   │   ├── particle_001.mrc
│   │   └── ...
│   └── test/
│       └── ...
```

**文件格式要求：**
- 每个文件是一个独立的 3D 颗粒数据
- 支持格式：`.mrc`, `.em`, `.h5`, `.hdf5`, `.nii`, `.nii.gz`
- **不需要标注文件**，每个文件本身就是颗粒
- 文件大小可以不同（会自动处理）

**使用 subtomogram 数据的优势：**
- ✅ 数据准备更简单（不需要标注文件）
- ✅ 训练更快（不需要从完整 tomogram 中提取 patches）
- ✅ 适合已经完成粒子提取的数据集

### 2.4 创建数据目录
```bash
# 创建数据目录结构
mkdir -p data/processed/{train,val,test}

# 将你的数据文件放入对应目录
# 选项 A：放入 tomogram 文件和对应的 .json 标注文件
# 选项 B：直接放入已提取的颗粒文件（.mrc, .h5 等）
```

## 3. 运行训练 / Running Training

### 3.1 基本训练命令

**所有参数都在 YAML 配置文件中管理，只需指定配置文件：**

```bash
python scripts/train.py --config config/default_config.yaml
```

就是这么简单！所有训练参数都在 `config/default_config.yaml` 文件中配置。

### 3.2 配置文件说明

在 YAML 配置文件中，你可以设置所有参数：

**数据相关参数：**
- `data.data_dir`: 数据目录路径
- `data.use_subtomograms`: 是否使用已提取的颗粒数据（true/false）
- `data.subtomogram.mask_type`: mask 类型（"full" 或 "center"）
- `data.subtomogram.mask_radius`: mask 半径（null=自动，0-1=相对值，>1=绝对值）

**模型相关参数：**
- `model.type`: 模型类型（"unet3d", "resnet3d", "transformer3d", "ensemble"）
- `model.batch_size`: 批次大小
- `model.learning_rate`: 学习率
- `model.num_epochs`: 训练轮数

**训练相关参数：**
- `training.gpus`: GPU 数量（0=CPU）
- `training.precision`: 精度（16 或 32）
- `training.seed`: 随机种子
- `training.num_workers`: 数据加载进程数

**实验跟踪：**
- `experiment.name`: 实验名称
- `experiment.wandb.enabled`: 是否启用 Weights & Biases

**Mask 类型说明：**
- `mask_type: "full"`：全体积 mask（整个体积都是颗粒）
- `mask_type: "center"`：中心区域 mask（只有中心区域是颗粒）
  - `mask_radius: null`：自动使用 `min(尺寸) // 4`
  - `mask_radius: 0.3`：相对值，表示最小尺寸的 30%
  - `mask_radius: 15`：绝对值，表示半径 15 像素

### 3.3 修改配置文件示例

**使用已提取的 Subtomogram 颗粒：**
编辑 `config/default_config.yaml`：
```yaml
data:
  data_dir: "data"
  use_subtomograms: true  # 启用 subtomogram 模式
  subtomogram:
    mask_type: "full"     # 或 "center"
    mask_radius: null     # 或 0.3 (相对值) 或 15 (绝对值)

model:
  type: "unet3d"
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 100

training:
  gpus: 1
  precision: 32
  seed: 42
```

然后运行：
```bash
python scripts/train.py --config config/default_config.yaml
```

### 3.4 快速测试运行

编辑配置文件，设置：
```yaml
training:
  fast_dev_run: true  # 快速测试模式
```

### 3.5 从检查点恢复训练

编辑配置文件，设置：
```yaml
training:
  resume_from_checkpoint: "experiments/results/checkpoints/last.ckpt"
```

## 4. 配置文件说明 / Configuration

### 4.1 配置文件结构
`config/default_config.yaml` 包含所有训练参数，分为以下几个部分：

- **data**: 数据相关设置（路径、格式、subtomogram 设置等）
- **model**: 模型相关设置（类型、架构、超参数等）
- **training**: 训练相关设置（GPU、精度、随机种子等）
- **experiment**: 实验跟踪设置（W&B、TensorBoard 等）
- **paths**: 路径设置（数据目录、结果目录等）

### 4.2 创建自定义配置文件
你可以复制默认配置文件并修改：
```bash
cp config/default_config.yaml config/my_config.yaml
# 编辑 my_config.yaml 修改参数
python scripts/train.py --config config/my_config.yaml
```

## 5. 训练输出 / Training Output

训练过程中会生成：
- **检查点文件**：`experiments/results/checkpoints/`
- **日志文件**：`experiments/results/logs/`
- **TensorBoard 日志**：可在 `experiments/results/logs/` 查看
- **Weights & Biases 日志**：如果启用了 W&B

### 5.1 查看训练进度
```bash
# 使用 TensorBoard
tensorboard --logdir experiments/results/logs

# 然后在浏览器中打开 http://localhost:6006
```

## 6. 常见问题 / Troubleshooting

### 6.0 Mask 类型选择问题
- **问题**：不知道应该选择 `full` 还是 `center` mask，以及如何设置半径
- **解决**：
  - **选择 `full`（默认）**：如果你的 subtomogram 文件已经是精确提取的颗粒，整个体积都是颗粒，没有背景
  - **选择 `center`**：如果你的 subtomogram 文件包含颗粒和周围背景，只有中心区域是真正的颗粒
  - **设置半径**：使用 `--mask-radius` 参数
    - 不指定：自动使用 `min(尺寸) // 4`
    - 相对值：`--mask-radius 0.3` 表示最小尺寸的 30%
    - 绝对值：`--mask-radius 15` 表示半径 15 像素
  - **建议**：如果不确定，先使用 `full`（默认值），这是最常见的场景

### 6.1 数据加载错误
- **问题**：找不到数据文件
- **解决**：检查数据目录结构是否正确，确保 `data/processed/{train,val,test}/` 下有数据文件

### 6.2 内存不足
- **问题**：CUDA out of memory
- **解决**：
  - 减小 `--batch-size`
  - 减小 `patch_size`（在配置文件中）
  - 使用 `--precision 16`（混合精度训练）

### 6.3 模块导入错误
- **问题**：`ModuleNotFoundError`
- **解决**：
  - 确保已安装所有依赖：`pip install -r requirements.txt`
  - 确保在项目根目录运行脚本
  - 检查 Python 路径是否正确

### 6.4 GPU 不可用
- **问题**：训练很慢，GPU 未使用
- **解决**：
  - 检查 CUDA 安装：`python -c "import torch; print(torch.cuda.is_available())"`
  - 确保安装了 GPU 版本的 PyTorch
  - 检查 `--gpus` 参数设置

## 7. 下一步 / Next Steps

训练完成后，你可以：
1. **评估模型**：使用 `scripts/evaluate.py`（如果存在）
2. **进行预测**：使用 `scripts/predict.py`（如果存在）
3. **分析结果**：查看 TensorBoard 或 W&B 日志
4. **调整超参数**：修改配置文件并重新训练

## 8. 示例命令总结 / Command Summary

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
# 选项 A：完整 tomogram + 标注文件
# 选项 B：已提取的 subtomogram 颗粒（推荐）⭐

# 3. 编辑配置文件（config/default_config.yaml）
#    设置所有参数：数据路径、模型类型、训练参数等

# 4. 开始训练（只需一个命令！）
python scripts/train.py --config config/default_config.yaml

# 5. 使用自定义配置文件
python scripts/train.py --config config/my_config.yaml

# 6. 查看训练日志
tensorboard --logdir experiments/results/logs
```

**就是这么简单！所有参数都在 YAML 配置文件中统一管理。**

