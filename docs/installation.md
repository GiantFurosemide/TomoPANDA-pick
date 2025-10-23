# Installation Guide

## 安装指南

## Prerequisites

## 环境要求

- Python 3.10 or higher
- CUDA 11.0+ (recommended for GPU acceleration)
- 16GB+ RAM (recommended)
- 50GB+ free disk space

- Python 3.10 或更高版本
- CUDA 11.0+ (推荐用于GPU加速)
- 16GB+ RAM (推荐)
- 50GB+ 可用磁盘空间

## Installation Steps

## 安装步骤

### 1. Clone the Repository

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/TomoPANDA-pick.git
cd TomoPANDA-pick
```

### 2. Create Virtual Environment

### 2. 创建虚拟环境

#### Using Conda (Recommended)

#### 使用 Conda (推荐)

```bash
# Create conda environment
conda create -n tomopanda python=3.10
conda activate tomopanda

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Using venv

#### 使用 venv

```bash
# Create virtual environment
python -m venv tomopanda-env
source tomopanda-env/bin/activate  # On Windows: tomopanda-env\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies

### 3. 安装依赖

```bash
# Install project dependencies
pip install -r requirements.txt

# Install project in development mode
pip install -e .
```

### 4. Verify Installation

### 4. 验证安装

```bash
# Test installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run tests
pytest tests/ -v
```

## Optional Dependencies

## 可选依赖

### For Development

### 开发环境

```bash
pip install -e ".[dev]"
```

### For Documentation

### 文档生成

```bash
pip install -e ".[docs]"
```

### For Jupyter Notebooks

### Jupyter 笔记本

```bash
pip install -e ".[notebooks]"
```

## Troubleshooting

## 故障排除

### Common Issues

### 常见问题

#### CUDA Issues

#### CUDA 问题

If you encounter CUDA-related issues:

如果遇到CUDA相关问题：

```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues

#### 内存问题

If you encounter out-of-memory errors:

如果遇到内存不足错误：

1. Reduce batch size in configuration
2. Use gradient accumulation
3. Enable mixed precision training

1. 在配置中减少批次大小
2. 使用梯度累积
3. 启用混合精度训练

#### Import Errors

#### 导入错误

If you encounter import errors:

如果遇到导入错误：

```bash
# Make sure you're in the project directory
cd TomoPANDA-pick

# Install in development mode
pip install -e .

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Getting Help

### 获取帮助

If you encounter issues not covered here:

如果遇到此处未涵盖的问题：

1. Check the [Issues](https://github.com/your-username/TomoPANDA-pick/issues) page
2. Create a new issue with detailed error information
3. Join our community discussions

1. 查看 [Issues](https://github.com/your-username/TomoPANDA-pick/issues) 页面
2. 创建包含详细错误信息的新问题
3. 加入我们的社区讨论
