#!/usr/bin/env python3
"""
2D Projection 模块使用示例

展示如何使用 2d_projection 模块的功能：
1. Fibonacci 球面均匀方向采样
2. 方向向量到 RELION ZYZ 欧拉角转换
3. 3D 体积到 2D 投影（带/不带 mask）
4. 完整的投影流程
"""

import os
import sys
import numpy as np
import mrcfile
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入 2d_projection 模块（使用 importlib 处理数字开头的模块名）
import importlib.util

# 导入 math_utils
math_utils_path = project_root / 'utils' / '2d_projection' / 'math_utils.py'
spec = importlib.util.spec_from_file_location("math_utils", str(math_utils_path))
math_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_utils)
fibonacci_sphere_sampling = math_utils.fibonacci_sphere_sampling
direction_to_relion_zyz = math_utils.direction_to_relion_zyz
zyz_euler_to_rotation_matrix = math_utils.zyz_euler_to_rotation_matrix

# 导入 projection
projection_path = project_root / 'utils' / '2d_projection' / 'projection.py'
spec = importlib.util.spec_from_file_location("projection", str(projection_path))
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)
project_3d_to_2d_rotated = projection.project_3d_to_2d_rotated


def create_sample_volume(size=(64, 64, 64), center=None, radius=15):
    """
    创建示例 3D 体积（球体）
    
    Parameters
    ----------
    size : tuple
        体积尺寸 (Z, Y, X)
    center : tuple or None
        球心位置，如果为 None 则使用体积中心
    radius : float
        球体半径
        
    Returns
    -------
    np.ndarray
        3D 体积数组
    """
    z, y, x = np.meshgrid(
        np.arange(size[0]),
        np.arange(size[1]),
        np.arange(size[2]),
        indexing='ij'
    )
    
    if center is None:
        center = (size[0] // 2, size[1] // 2, size[2] // 2)
    
    # 计算距离
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    
    # 创建球体（带一些噪声）
    volume = np.exp(-dist**2 / (2 * (radius/3)**2))
    volume += np.random.normal(0, 0.1, size=size)
    volume = np.clip(volume, 0, 1)
    
    return volume.astype(np.float32)


def create_spherical_mask(size=(64, 64, 64), center=None, radius=20):
    """
    创建球形 mask
    
    Parameters
    ----------
    size : tuple
        Mask 尺寸
    center : tuple or None
        球心位置
    radius : float
        球体半径
        
    Returns
    -------
    np.ndarray
        二值 mask（True 表示 mask 内）
    """
    z, y, x = np.meshgrid(
        np.arange(size[0]),
        np.arange(size[1]),
        np.arange(size[2]),
        indexing='ij'
    )
    
    if center is None:
        center = (size[0] // 2, size[1] // 2, size[2] // 2)
    
    dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
    mask = dist <= radius
    
    return mask


def demonstrate_fibonacci_sampling():
    """演示 Fibonacci 球面采样"""
    print("\n=== 1. Fibonacci 球面均匀方向采样 ===\n")
    
    # 生成 10 个均匀分布的方向
    n_orientations = 10
    directions = fibonacci_sphere_sampling(n_orientations)
    
    print(f"生成了 {n_orientations} 个方向向量：")
    print(f"形状: {directions.shape}")
    print(f"\n前 5 个方向向量：")
    for i in range(min(5, n_orientations)):
        v = directions[i]
        print(f"  方向 {i}: ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")
        print(f"    模长: {np.linalg.norm(v):.6f}")  # 应该是 1.0
    
    # 验证方向向量是单位向量
    norms = np.linalg.norm(directions, axis=1)
    print(f"\n方向向量模长范围: [{norms.min():.6f}, {norms.max():.6f}]")
    print(f"所有方向都是单位向量: {np.allclose(norms, 1.0)}")


def demonstrate_direction_to_euler():
    """演示方向向量到欧拉角转换"""
    print("\n=== 2. 方向向量到 RELION ZYZ 欧拉角转换 ===\n")
    
    # 生成一些测试方向
    test_directions = np.array([
        [0, 0, 1],    # 沿 Z 轴
        [1, 0, 0],    # 沿 X 轴
        [0, 1, 0],    # 沿 Y 轴
        [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]  # 对角线方向
    ])
    
    print("测试方向向量及其对应的 ZYZ 欧拉角：")
    for i, direction in enumerate(test_directions):
        angles = direction_to_relion_zyz(direction, psi_deg=0.0)
        print(f"\n方向 {i}: ({direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f})")
        print(f"  Rot: {angles[0]:.2f}°, Tilt: {angles[1]:.2f}°, Psi: {angles[2]:.2f}°")
    
    # 使用 Fibonacci 采样生成方向并转换
    print("\n使用 Fibonacci 采样生成 5 个方向：")
    directions = fibonacci_sphere_sampling(5)
    angles = direction_to_relion_zyz(directions, psi_deg=0.0)
    
    for i in range(5):
        print(f"  方向 {i}: Rot={angles[i, 0]:.2f}°, Tilt={angles[i, 1]:.2f}°, Psi={angles[i, 2]:.2f}°")


def demonstrate_projection_without_mask():
    """演示不带 mask 的投影"""
    print("\n=== 3. 3D 到 2D 投影（不带 mask）===\n")
    
    # 创建示例体积
    volume = create_sample_volume(size=(64, 64, 64), radius=20)
    print(f"创建了示例体积，形状: {volume.shape}")
    print(f"数值范围: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # 定义投影参数
    rot_deg = 45.0
    tilt_deg = 30.0
    psi_deg = 0.0
    box_size = 64
    
    print(f"\n投影参数:")
    print(f"  Rot: {rot_deg}°, Tilt: {tilt_deg}°, Psi: {psi_deg}°")
    print(f"  输出尺寸: {box_size}x{box_size}")
    
    # 测试不同的投影模式
    modes = ["sum", "mean", "max", "central_slice"]
    
    for mode in modes:
        projection = project_3d_to_2d_rotated(
            volume=volume,
            rot_deg=rot_deg,
            tilt_deg=tilt_deg,
            psi_deg=psi_deg,
            box_size=box_size,
            mode=mode,
            normalize=True
        )
        print(f"\n{mode} 模式:")
        print(f"  投影形状: {projection.shape}")
        print(f"  数值范围: [{projection.min():.3f}, {projection.max():.3f}]")
        print(f"  均值: {projection.mean():.3f}, 标准差: {projection.std():.3f}")


def demonstrate_projection_with_mask():
    """演示带 mask 的投影"""
    print("\n=== 4. 3D 到 2D 投影（带 mask）===\n")
    
    # 创建示例体积和 mask
    volume = create_sample_volume(size=(64, 64, 64), radius=20)
    mask = create_spherical_mask(size=(64, 64, 64), radius=18)
    
    print(f"体积形状: {volume.shape}")
    print(f"Mask 形状: {mask.shape}")
    print(f"Mask 内体素数: {np.sum(mask)} ({np.sum(mask) / mask.size * 100:.1f}%)")
    print(f"Mask 外体素数: {np.sum(~mask)} ({np.sum(~mask) / mask.size * 100:.1f}%)")
    
    # 计算 mask 外区域的标准差（用于噪声）
    masked_out = volume[~mask]
    noise_std = np.std(masked_out) if len(masked_out) > 0 else 0.1
    print(f"Mask 外区域标准差: {noise_std:.3f}")
    
    # 投影参数
    rot_deg = 45.0
    tilt_deg = 30.0
    psi_deg = 0.0
    box_size = 64
    
    # 不带 mask 的投影
    projection_no_mask = project_3d_to_2d_rotated(
        volume=volume,
        rot_deg=rot_deg,
        tilt_deg=tilt_deg,
        psi_deg=psi_deg,
        box_size=box_size,
        mode="sum",
        normalize=True
    )
    
    # 带 mask 的投影
    projection_with_mask = project_3d_to_2d_rotated(
        volume=volume,
        rot_deg=rot_deg,
        tilt_deg=tilt_deg,
        psi_deg=psi_deg,
        box_size=box_size,
        mode="sum",
        normalize=True,
        mask=mask,
        noise_mean=0.0,
        noise_std=noise_std
    )
    
    print(f"\n不带 mask 的投影:")
    print(f"  形状: {projection_no_mask.shape}")
    print(f"  数值范围: [{projection_no_mask.min():.3f}, {projection_no_mask.max():.3f}]")
    
    print(f"\n带 mask 的投影:")
    print(f"  形状: {projection_with_mask.shape}")
    print(f"  数值范围: [{projection_with_mask.min():.3f}, {projection_with_mask.max():.3f}]")
    
    # 计算差异
    diff = np.abs(projection_with_mask - projection_no_mask)
    print(f"\n差异统计:")
    print(f"  平均差异: {diff.mean():.3f}")
    print(f"  最大差异: {diff.max():.3f}")


def demonstrate_multiple_orientations():
    """演示多个方向的投影"""
    print("\n=== 5. 多个方向的投影 ===\n")
    
    # 创建示例体积
    volume = create_sample_volume(size=(64, 64, 64), radius=20)
    
    # 生成 5 个 Fibonacci 方向
    n_orientations = 5
    directions = fibonacci_sphere_sampling(n_orientations)
    angles = direction_to_relion_zyz(directions, psi_deg=0.0)
    
    print(f"生成了 {n_orientations} 个方向的投影：")
    
    projections = []
    for i in range(n_orientations):
        rot_deg = angles[i, 0]
        tilt_deg = angles[i, 1]
        psi_deg = angles[i, 2]
        
        projection = project_3d_to_2d_rotated(
            volume=volume,
            rot_deg=rot_deg,
            tilt_deg=tilt_deg,
            psi_deg=psi_deg,
            box_size=64,
            mode="sum",
            normalize=True
        )
        projections.append(projection)
        
        print(f"\n方向 {i}:")
        print(f"  方向向量: ({directions[i, 0]:.3f}, {directions[i, 1]:.3f}, {directions[i, 2]:.3f})")
        print(f"  欧拉角: Rot={rot_deg:.2f}°, Tilt={tilt_deg:.2f}°, Psi={psi_deg:.2f}°")
        print(f"  投影均值: {projection.mean():.3f}, 标准差: {projection.std():.3f}")
    
    # 保存投影到临时文件（可选）
    print(f"\n所有投影已生成，共 {len(projections)} 个")


def demonstrate_save_projection_stack():
    """演示保存投影栈到 MRC 文件"""
    print("\n=== 6. 保存投影栈到 MRC 文件 ===\n")
    
    # 创建示例体积
    volume = create_sample_volume(size=(64, 64, 64), radius=20)
    
    # 生成 3 个方向的投影
    n_orientations = 3
    directions = fibonacci_sphere_sampling(n_orientations)
    angles = direction_to_relion_zyz(directions, psi_deg=0.0)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    stack_path = os.path.join(temp_dir, "projections.mrcs")
    
    try:
        # 生成投影栈
        projections = []
        for i in range(n_orientations):
            projection = project_3d_to_2d_rotated(
                volume=volume,
                rot_deg=angles[i, 0],
                tilt_deg=angles[i, 1],
                psi_deg=angles[i, 2],
                box_size=64,
                mode="sum",
                normalize=True
            )
            projections.append(projection)
        
        # 转换为栈格式 (N, H, W)
        stack_data = np.stack(projections, axis=0).astype(np.float32)
        print(f"投影栈形状: {stack_data.shape}")
        
        # 保存为 MRC 栈文件
        with mrcfile.new(stack_path, overwrite=True) as mrc:
            mrc.set_data(stack_data)
            mrc.update_header_from_data()
        
        print(f"已保存投影栈到: {stack_path}")
        
        # 验证文件
        with mrcfile.open(stack_path) as mrc:
            loaded_data = mrc.data
            print(f"从文件读取的形状: {loaded_data.shape}")
            print(f"数据匹配: {np.allclose(stack_data, loaded_data)}")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print("已清理临时文件")


def demonstrate_rotation_matrix():
    """演示旋转矩阵的使用"""
    print("\n=== 7. ZYZ 欧拉角到旋转矩阵转换 ===\n")
    
    # 测试一些角度
    test_angles = [
        (0, 0, 0),
        (45, 30, 0),
        (90, 45, 0),
    ]
    
    print("欧拉角到旋转矩阵转换：")
    for rot, tilt, psi in test_angles:
        R = zyz_euler_to_rotation_matrix(rot, tilt, psi)
        print(f"\n欧拉角: Rot={rot}°, Tilt={tilt}°, Psi={psi}°")
        print(f"旋转矩阵:")
        print(f"  {R[0]}")
        print(f"  {R[1]}")
        print(f"  {R[2]}")
        
        # 验证旋转矩阵的性质（应该是正交矩阵）
        det = np.linalg.det(R)
        print(f"  行列式: {det:.6f} (应该接近 1.0)")
        print(f"  是否正交: {np.allclose(R @ R.T, np.eye(3))}")


def main():
    """主函数 - 运行所有演示"""
    print("=" * 60)
    print("2D Projection 模块使用示例")
    print("=" * 60)
    
    try:
        # 运行各个演示
        demonstrate_fibonacci_sampling()
        demonstrate_direction_to_euler()
        demonstrate_projection_without_mask()
        demonstrate_projection_with_mask()
        demonstrate_multiple_orientations()
        demonstrate_save_projection_stack()
        demonstrate_rotation_matrix()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        print("\n使用提示：")
        print("1. 使用 fibonacci_sphere_sampling() 生成均匀分布的方向")
        print("2. 使用 direction_to_relion_zyz() 转换为 RELION 欧拉角")
        print("3. 使用 project_3d_to_2d_rotated() 进行投影")
        print("4. 可以使用 mask 参数来保留特定区域的信号")
        print("5. 运行完整流程：python utils/2d_projection/main.py -i config.yaml")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请确保已安装所有依赖：")
        print("pip install -r requirements.txt")


if __name__ == '__main__':
    main()

