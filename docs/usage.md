# Usage Guide

## 使用指南

## Quick Start

## 快速开始

### 1. Prepare Your Data

### 1. 准备数据

```bash
# Organize your data in the following structure:
# 按以下结构组织数据：

data/
├── raw/
│   ├── tomograms/          # Your tomogram files (.mrc, .em, .h5)
│   └── annotations/        # Particle annotations (.json)
└── processed/             # Will be created automatically
    ├── train/
    ├── val/
    └── test/
```

### 2. Prepare Data

### 2. 数据准备

```bash
# Run data preparation script
python scripts/data_preparation.py --config config/default_config.yaml
```

### 3. Train a Model

### 3. 训练模型

```bash
# Train UNet3D model
python scripts/train.py --model unet3d --config config/default_config.yaml

# Train ResNet3D model
python scripts/train.py --model resnet3d --config config/model_configs/resnet3d_config.yaml

# Train Transformer3D model
python scripts/train.py --model transformer3d --config config/model_configs/transformer3d_config.yaml
```

### 4. Evaluate Model

### 4. 评估模型

```bash
# Evaluate trained model
python scripts/evaluate.py --model_path experiments/results/checkpoints/best_model.pth
```

### 5. Make Predictions

### 5. 进行预测

```bash
# Predict on new data
python scripts/predict.py --model_path experiments/results/checkpoints/best_model.pth --input data/test/
```

## Advanced Usage

## 高级用法

### Custom Configuration

### 自定义配置

Create your own configuration file:

创建自己的配置文件：

```yaml
# my_config.yaml
model:
  type: "unet3d"
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 200

data:
  preprocessing:
    patch_size: [128, 128, 128]
    overlap: 0.25

training:
  early_stopping:
    patience: 20
```

Use your configuration:

使用你的配置：

```bash
python scripts/train.py --config my_config.yaml
```

### Experiment Tracking

### 实验跟踪

#### Weights & Biases

```bash
# Login to wandb
wandb login

# Train with wandb logging
python scripts/train.py --model unet3d --experiment-name "my_experiment"
```

#### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir experiments/results/logs

# View in browser: http://localhost:6006
```

### Hyperparameter Search

### 超参数搜索

```bash
# Run hyperparameter search
python scripts/hyperparameter_search.py --model unet3d --config config/default_config.yaml
```

### Multi-GPU Training

### 多GPU训练

```bash
# Train with multiple GPUs
python scripts/train.py --model unet3d --gpus 2
```

### Mixed Precision Training

### 混合精度训练

```bash
# Train with mixed precision
python scripts/train.py --model unet3d --precision 16
```

## API Usage

## API 使用

### Programmatic Usage

### 编程使用

```python
import torch
from models import ModelFactory
from data.utils import CryoETDataLoader

# Create model
model = ModelFactory.create_model(
    model_type='unet3d',
    in_channels=1,
    num_classes=2,
    base_channels=64
)

# Create data loader
dataloaders = CryoETDataLoader.create_dataloaders(
    data_dir='data',
    batch_size=4
)

# Train model
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, dataloaders['train'], dataloaders['val'])
```

### Custom Model

### 自定义模型

```python
from models.base.base_model import BaseModel
import torch.nn as nn

class MyCustomModel(BaseModel):
    def _build_model(self):
        self.model = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, self.num_classes, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Use your custom model
model = MyCustomModel(in_channels=1, num_classes=2)
```

## Data Formats

## 数据格式

### Supported Input Formats

### 支持的输入格式

- **MRC files** (.mrc): Standard cryoEM format
- **EM files** (.em): EMAN format
- **HDF5 files** (.h5, .hdf5): Hierarchical data format
- **NIfTI files** (.nii, .nii.gz): Medical imaging format

- **MRC 文件** (.mrc): 标准 cryoEM 格式
- **EM 文件** (.em): EMAN 格式
- **HDF5 文件** (.h5, .hdf5): 分层数据格式
- **NIfTI 文件** (.nii, .nii.gz): 医学成像格式

### Annotation Format

### 标注格式

Particle annotations should be in JSON format:

粒子标注应为JSON格式：

```json
{
  "particles": [
    {
      "center": [x, y, z],
      "radius": 5,
      "confidence": 0.95
    }
  ]
}
```

## Performance Tips

## 性能提示

### Memory Optimization

### 内存优化

1. **Reduce batch size**: Use smaller batches if you run out of memory
2. **Use gradient accumulation**: Accumulate gradients over multiple mini-batches
3. **Enable mixed precision**: Use 16-bit precision for training

1. **减少批次大小**: 如果内存不足，使用更小的批次
2. **使用梯度累积**: 在多个小批次上累积梯度
3. **启用混合精度**: 使用16位精度进行训练

### Speed Optimization

### 速度优化

1. **Use multiple workers**: Increase `num_workers` for data loading
2. **Pin memory**: Enable `pin_memory=True` for faster GPU transfer
3. **Use SSD storage**: Store data on SSD for faster I/O

1. **使用多个工作进程**: 增加数据加载的 `num_workers`
2. **固定内存**: 启用 `pin_memory=True` 以加快GPU传输
3. **使用SSD存储**: 将数据存储在SSD上以加快I/O

## Troubleshooting

## 故障排除

### Common Issues

### 常见问题

#### Out of Memory

#### 内存不足

```bash
# Reduce batch size
python scripts/train.py --batch-size 2

# Use gradient accumulation
# Add to config: accumulate_grad_batches: 4
```

#### Slow Training

#### 训练缓慢

```bash
# Increase number of workers
python scripts/train.py --num-workers 8

# Use mixed precision
python scripts/train.py --precision 16
```

#### Poor Performance

#### 性能不佳

1. **Check data quality**: Ensure annotations are accurate
2. **Adjust model architecture**: Try different model types
3. **Tune hyperparameters**: Use hyperparameter search
4. **Increase training time**: Train for more epochs

1. **检查数据质量**: 确保标注准确
2. **调整模型架构**: 尝试不同的模型类型
3. **调整超参数**: 使用超参数搜索
4. **增加训练时间**: 训练更多轮次

### Getting Help

### 获取帮助

- **Documentation**: Check the docs/ directory
- **Issues**: Report bugs on GitHub
- **Discussions**: Join community discussions
- **Email**: Contact the maintainers

- **文档**: 查看 docs/ 目录
- **问题**: 在GitHub上报告错误
- **讨论**: 加入社区讨论
- **邮件**: 联系维护者
