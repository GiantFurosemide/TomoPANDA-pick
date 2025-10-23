# TomoPANDA-pick

A deep learning framework for 3D particle picking in cryo-electron tomography (cryoET) data.

深度学习框架，用于冷冻电子断层扫描 (cryoET) 数据中的 3D 粒子挑选。

## Project Overview

TomoPANDA-pick is a specialized deep learning framework for 3D particle picking in cryoET data. This project aims to test and compare the performance of different deep learning models in 3D particle detection and localization tasks.

## 项目概述

TomoPANDA-pick 是一个专门用于 cryoET 数据中 3D 粒子挑选的深度学习框架。该项目旨在测试和比较不同的深度学习模型在 3D 粒子检测和定位任务中的性能。

## Project Architecture

## 项目架构

```
TomoPANDA-pick/
├── README.md                    # Project documentation / 项目说明文档
├── requirements.txt             # Python dependencies / Python 依赖包
├── setup.py                     # Project installation configuration / 项目安装配置
├── config/                      # Configuration files directory / 配置文件目录
│   ├── __init__.py
│   ├── default_config.yaml      # Default configuration / 默认配置
│   └── model_configs/           # Model-specific configurations / 模型特定配置
│       ├── unet3d_config.yaml
│       ├── resnet3d_config.yaml
│       └── transformer3d_config.yaml
├── data/                        # Data directory / 数据目录
│   ├── raw/                     # Raw data / 原始数据
│   │   ├── tomograms/          # Tomography data / 断层扫描数据
│   │   └── annotations/         # Annotation data / 标注数据
│   ├── processed/              # Processed data / 预处理数据
│   │   ├── train/              # Training set / 训练集
│   │   ├── val/                # Validation set / 验证集
│   │   └── test/               # Test set / 测试集
│   └── utils/                   # Data utilities / 数据工具
│       ├── __init__.py
│       ├── data_loader.py      # Data loader / 数据加载器
│       ├── preprocessing.py    # Data preprocessing / 数据预处理
│       └── augmentation.py     # Data augmentation / 数据增强
├── models/                      # Model definitions / 模型定义
│   ├── __init__.py
│   ├── base/                   # Base model classes / 基础模型类
│   │   ├── __init__.py
│   │   └── base_model.py
│   ├── unet3d/                 # 3D U-Net models / 3D U-Net 模型
│   │   ├── __init__.py
│   │   ├── unet3d.py
│   │   └── unet3d_plus.py
│   ├── resnet3d/               # 3D ResNet models / 3D ResNet 模型
│   │   ├── __init__.py
│   │   ├── resnet3d.py
│   │   └── resnet3d_plus.py
│   ├── transformer3d/          # 3D Transformer models / 3D Transformer 模型
│   │   ├── __init__.py
│   │   ├── transformer3d.py
│   │   └── attention3d.py
│   └── ensemble/               # Ensemble models / 集成模型
│       ├── __init__.py
│       └── ensemble_model.py
├── training/                    # Training related / 训练相关
│   ├── __init__.py
│   ├── trainer.py              # Trainer / 训练器
│   ├── loss_functions.py       # Loss functions / 损失函数
│   ├── metrics.py              # Evaluation metrics / 评估指标
│   └── callbacks.py            # Training callbacks / 训练回调
├── inference/                   # Inference related / 推理相关
│   ├── __init__.py
│   ├── predictor.py            # Predictor / 预测器
│   ├── postprocessing.py       # Post-processing / 后处理
│   └── visualization.py        # Visualization / 可视化
├── experiments/                # Experiment management / 实验管理
│   ├── __init__.py
│   ├── experiment_manager.py   # Experiment manager / 实验管理器
│   ├── hyperparameter_search.py # Hyperparameter search / 超参数搜索
│   └── results/                # Experiment results / 实验结果
│       ├── logs/               # Training logs / 训练日志
│       ├── checkpoints/        # Model checkpoints / 模型检查点
│       └── visualizations/     # Result visualizations / 结果可视化
├── utils/                       # Utility functions / 工具函数
│   ├── __init__.py
│   ├── io_utils.py             # I/O utilities / I/O 工具
│   ├── math_utils.py           # Math utilities / 数学工具
│   ├── visualization_utils.py  # Visualization utilities / 可视化工具
│   └── logging_utils.py        # Logging utilities / 日志工具
├── scripts/                     # Script files / 脚本文件
│   ├── train.py                # Training script / 训练脚本
│   ├── evaluate.py             # Evaluation script / 评估脚本
│   ├── predict.py              # Prediction script / 预测脚本
│   ├── data_preparation.py     # Data preparation script / 数据准备脚本
│   └── benchmark.py            # Benchmark script / 基准测试脚本
├── tests/                       # Test files / 测试文件
│   ├── __init__.py
│   ├── test_models.py          # Model tests / 模型测试
│   ├── test_data_loader.py     # Data loader tests / 数据加载器测试
│   └── test_training.py        # Training tests / 训练测试
├── docs/                        # Documentation / 文档
│   ├── installation.md         # Installation guide / 安装说明
│   ├── usage.md                # Usage guide / 使用说明
│   ├── model_architecture.md   # Model architecture guide / 模型架构说明
│   └── api_reference.md        # API reference / API 参考
└── notebooks/                   # Jupyter notebooks / Jupyter 笔记本
    ├── data_exploration.ipynb  # Data exploration / 数据探索
    ├── model_comparison.ipynb  # Model comparison / 模型比较
    └── results_analysis.ipynb # Results analysis / 结果分析
```

## Core Features / 核心功能

### 1. Data Management / 数据管理
- **Data Loading / 数据加载**: Support for multiple cryoET data formats (MRC, EM, HDF5)
- **Data Preprocessing / 数据预处理**: Noise removal, contrast enhancement, normalization
- **Data Augmentation / 数据增强**: 3D rotation, flipping, elastic deformation, etc.
- **Data Splitting / 数据分割**: Automatic train/validation/test set division

### 2. Model Architecture / 模型架构
- **3D U-Net**: Classic 3D segmentation network / 经典的 3D 分割网络
- **3D ResNet**: Residual networks for 3D feature extraction / 残差网络用于 3D 特征提取
- **3D Transformer**: Attention-based 3D models / 基于注意力机制的 3D 模型
- **Ensemble Models**: Multi-model fusion for improved performance / 多模型融合提升性能

### 3. Training and Evaluation / 训练与评估
- **Loss Functions / 损失函数**: Dice Loss, Focal Loss, Combined losses
- **Evaluation Metrics / 评估指标**: Precision, Recall, F1-Score, IoU
- **Training Strategies / 训练策略**: Learning rate scheduling, early stopping, model checkpointing
- **Hyperparameter Optimization / 超参数优化**: Grid search, Bayesian optimization

### 4. Inference and Post-processing / 推理与后处理
- **Batch Inference / 批量推理**: Efficient large-scale data processing
- **Post-processing / 后处理**: Non-maximum suppression, connected component analysis
- **Visualization / 可视化**: 3D rendering, result comparison display

## Technology Stack / 技术栈

- **Deep Learning Framework / 深度学习框架**: PyTorch, PyTorch Lightning
- **Data Processing / 数据处理**: NumPy, SciPy, OpenCV
- **Visualization / 可视化**: Matplotlib, Plotly, VTK
- **Experiment Management / 实验管理**: Weights & Biases, TensorBoard
- **Configuration Management / 配置管理**: YAML, Hydra
- **Testing / 测试**: pytest

## Quick Start / 快速开始

### Requirements / 环境要求
- Python 3.10+
- CUDA 11.0+ (recommended / 推荐)
- 16GB+ RAM (recommended / 推荐)

### Installation / 安装
```bash
# Clone the project / 克隆项目
git clone <repository-url>
cd TomoPANDA-pick

# Create virtual environment / 创建虚拟环境
conda create -n tomopanda-pick python=3.10
conda activate tomopanda-pick

# Install dependencies / 安装依赖
pip install -r requirements.txt
```

### Basic Usage / 基本使用
```bash
# Data preparation / 数据准备
python scripts/data_preparation.py --config config/default_config.yaml

# Train model / 训练模型
python scripts/train.py --model unet3d --config config/model_configs/unet3d_config.yaml

# Evaluate model / 评估模型
python scripts/evaluate.py --model_path experiments/results/checkpoints/best_model.pth

# Predict / 预测
python scripts/predict.py --model_path experiments/results/checkpoints/best_model.pth --input data/test/
```

## Development Roadmap / 开发计划

### Phase 1: Basic Framework (Current) / 基础框架 (当前)
- [x] Project structure design / 项目结构设计
- [ ] Basic model implementation / 基础模型实现
- [ ] Data loader / 数据加载器
- [ ] Training pipeline / 训练管道

### Phase 2: Model Implementation / 模型实现
- [ ] 3D U-Net implementation / 3D U-Net 实现
- [ ] 3D ResNet implementation / 3D ResNet 实现
- [ ] 3D Transformer implementation / 3D Transformer 实现
- [ ] Loss functions and evaluation metrics / 损失函数和评估指标

### Phase 3: Experiments and Optimization / 实验与优化
- [ ] Hyperparameter search / 超参数搜索
- [ ] Model comparison / 模型比较
- [ ] Performance optimization / 性能优化
- [ ] Results analysis / 结果分析

### Phase 4: Deployment and Documentation / 部署与文档
- [ ] Model deployment / 模型部署
- [ ] API interface / API 接口
- [ ] Complete documentation / 完整文档
- [ ] User guide / 用户指南

## Contributing / 贡献指南

1. Fork the project / Fork 项目
2. Create your feature branch (`git checkout -b feature/AmazingFeature`) / 创建特性分支
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`) / 提交更改
4. Push to the branch (`git push origin feature/AmazingFeature`) / 推送到分支
5. Open a Pull Request / 创建 Pull Request

## License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## Contact / 联系方式

- Project Maintainer: [Your Name] / 项目维护者: [您的姓名]
- Email: [Your Email] / 邮箱: [您的邮箱]
- Project Link: [Project Link] / 项目链接: [项目链接]

## Acknowledgments / 致谢

Thanks to all researchers and developers who have contributed to cryoET and deep learning research.

感谢所有为 cryoET 和深度学习研究做出贡献的研究者和开发者。
