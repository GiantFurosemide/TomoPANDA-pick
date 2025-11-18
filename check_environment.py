#!/usr/bin/env python3
"""
Environment check script for TomoPANDA-pick
检查训练环境是否配置正确
"""

import sys
import importlib

def check_python_version():
    """Check Python version"""
    print("=" * 60)
    print("1. 检查 Python 版本 / Checking Python version...")
    print("=" * 60)
    
    version = sys.version_info
    print(f"当前 Python 版本 / Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ 错误：需要 Python 3.10 或更高版本")
        print("   Error: Python 3.10+ is required")
        print("   请使用 python3 命令或创建虚拟环境")
        return False
    else:
        print("✓ Python 版本符合要求 / Python version OK")
        return True


def check_dependencies():
    """Check required dependencies"""
    print("\n" + "=" * 60)
    print("2. 检查依赖包 / Checking dependencies...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'pytorch_lightning': 'PyTorch Lightning',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'mrcfile': 'mrcfile',
        'h5py': 'h5py',
        'loguru': 'loguru',
    }
    
    optional_packages = {
        'wandb': 'Weights & Biases',
        'tensorboard': 'TensorBoard',
    }
    
    all_ok = True
    
    print("\n必需依赖 / Required dependencies:")
    for package, name in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: 未安装 / Not installed")
            all_ok = False
    
    print("\n可选依赖 / Optional dependencies:")
    for package, name in optional_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ⚠ {name}: 未安装（可选）/ Not installed (optional)")
    
    return all_ok


def check_project_modules():
    """Check if project modules can be imported"""
    print("\n" + "=" * 60)
    print("3. 检查项目模块 / Checking project modules...")
    print("=" * 60)
    
    modules_to_check = [
        ('models', 'ModelFactory'),
        ('data.utils', 'CryoETDataLoader'),
        ('training.loss_functions', 'DiceFocalLoss'),
        ('training.metrics', 'SegmentationMetrics'),
        ('training.callbacks', 'CryoETCallbacks'),
    ]
    
    all_ok = True
    
    for module_path, class_name in modules_to_check:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name, None)
            if cls is not None:
                print(f"  ✓ {module_path}.{class_name}")
            else:
                print(f"  ❌ {module_path}.{class_name}: 类不存在 / Class not found")
                all_ok = False
        except ImportError as e:
            print(f"  ❌ {module_path}: 导入失败 / Import failed - {e}")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "=" * 60)
    print("4. 检查 GPU / CUDA / Checking GPU/CUDA...")
    print("=" * 60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"  ✓ CUDA 可用 / CUDA available")
            print(f"  CUDA 版本 / CUDA version: {torch.version.cuda}")
            print(f"  GPU 数量 / Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  ⚠ CUDA 不可用，将使用 CPU 训练（较慢）")
            print("     CUDA not available, will use CPU (slower)")
        
        return True
    except ImportError:
        print("  ❌ PyTorch 未安装，无法检查 CUDA")
        print("     PyTorch not installed, cannot check CUDA")
        return False


def check_data_structure():
    """Check data directory structure"""
    print("\n" + "=" * 60)
    print("5. 检查数据目录结构 / Checking data directory structure...")
    print("=" * 60)
    
    import os
    from pathlib import Path
    
    data_dir = Path('data')
    processed_dir = data_dir / 'processed'
    
    if not data_dir.exists():
        print("  ⚠ data/ 目录不存在 / data/ directory does not exist")
        print("     请创建数据目录 / Please create data directory")
        return False
    
    splits = ['train', 'val', 'test']
    all_exist = True
    
    for split in splits:
        split_dir = processed_dir / split
        if split_dir.exists():
            files = list(split_dir.glob('*.mrc')) + list(split_dir.glob('*.h5'))
            print(f"  ✓ {split}/: {len(files)} 个文件 / {len(files)} files")
        else:
            print(f"  ⚠ {split}/: 目录不存在 / directory does not exist")
            all_exist = False
    
    if not all_exist:
        print("\n  提示：数据目录结构应该是：")
        print("  Hint: Data directory structure should be:")
        print("    data/processed/train/")
        print("    data/processed/val/")
        print("    data/processed/test/")
    
    return all_exist


def main():
    """Main check function"""
    print("\n" + "=" * 60)
    print("TomoPANDA-pick 环境检查 / Environment Check")
    print("=" * 60 + "\n")
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_project_modules(),
        check_cuda(),
        check_data_structure(),
    ]
    
    print("\n" + "=" * 60)
    print("检查结果总结 / Check Summary")
    print("=" * 60)
    
    if all(checks[:3]):  # Python, dependencies, and modules must pass
        print("\n✓ 基本环境检查通过！可以开始训练。")
        print("  ✓ Basic environment check passed! Ready to train.")
        print("\n运行训练命令 / Run training command:")
        print("  python scripts/train.py --model unet3d --data-dir data")
        return 0
    else:
        print("\n❌ 环境检查未通过，请先解决上述问题。")
        print("  ❌ Environment check failed, please fix the issues above.")
        print("\n安装依赖 / Install dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())

