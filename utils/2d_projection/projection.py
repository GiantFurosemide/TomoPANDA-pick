"""
3D 到 2D 投影模块

提供将 3D 体积投影到 2D 图像的功能，支持多种投影模式。
支持CPU和GPU两种模式，GPU模式使用PyTorch加速。
"""

import numpy as np
from typing import Literal, Optional
import os
import importlib.util

# 尝试导入PyTorch（可选，用于GPU加速）
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

# 导入 math_utils（处理数字开头的模块名）
math_utils_path = os.path.join(os.path.dirname(__file__), 'math_utils.py')
spec = importlib.util.spec_from_file_location("math_utils", math_utils_path)
math_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_utils)
zyz_euler_to_rotation_matrix = math_utils.zyz_euler_to_rotation_matrix


def project_3d_to_2d_rotated(
    volume: np.ndarray,
    rot_deg: float,
    tilt_deg: float,
    psi_deg: float,
    box_size: int,
    mode: Literal["sum", "mean", "max", "central_slice"] = "sum",
    normalize: bool = True,
    mask: np.ndarray = None,
    noise_mean: float = 0.0,
    noise_std: float = None
) -> np.ndarray:
    """
    将 3D 体积按照给定的 ZYZ 欧拉角旋转后投影到 2D（完整实现）。
    
    这个函数实现完整的 3D 旋转和投影流程。
    
    Parameters
    ----------
    volume : np.ndarray
        3D 体积数组，形状为 (Z, Y, X)
    rot_deg : float
        Rot 角度（度）
    tilt_deg : float
        Tilt 角度（度）
    psi_deg : float
        Psi 角度（度）
    box_size : int
        输出图像尺寸（box_size × box_size）
    mode : str, optional
        投影模式，默认 "sum"。可选值：
        - "sum": 沿Z轴求和投影（累加所有层，适合增强信号）
        - "mean": 沿Z轴平均投影（取平均值，适合减少噪声）
        - "max": 沿Z轴最大投影（取每列最大值，适合突出最强信号）
        - "central_slice": 中心切片（取Z方向中心层，类似单层图像）
    normalize : bool, optional
        是否归一化输出，默认 True
    mask : np.ndarray, optional
        3D 二值 mask，形状与 volume 相同。True/1 表示保留原始信号，False/0 表示用噪声填充。
        如果为 None，则不应用 mask。
    noise_mean : float, optional
        Gaussian 噪声的均值，默认 0.0
    noise_std : float, optional
        Gaussian 噪声的标准差。如果为 None，则使用 volume 在 mask 外区域的标准差。
        
    Returns
    -------
    np.ndarray
        2D 投影图像，形状为 (box_size, box_size)
    """
    # 验证输入 volume 的维度
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(
            f"volume must be 3D array, but got {volume.ndim}D array with shape {volume.shape}. "
            f"Expected shape: (Z, Y, X)"
        )
    
    # 应用 mask（如果提供）
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != volume.shape:
            raise ValueError(f"mask shape {mask.shape} must match volume shape {volume.shape}")
        
        # 计算噪声标准差（如果未提供）
        if noise_std is None:
            # 使用 mask 外区域的标准差
            masked_out = volume[~mask]
            if len(masked_out) > 0:
                noise_std = np.std(masked_out)
            else:
                # 如果 mask 外没有数据，使用整个 volume 的标准差
                noise_std = np.std(volume)
        
        # 创建带噪声的 volume
        volume_masked = volume.copy()
        # 在 mask 外区域填充 Gaussian 噪声
        noise = np.random.normal(noise_mean, noise_std, size=volume.shape)
        volume_masked[~mask] = noise[~mask]
        volume = volume_masked
    
    # 获取旋转矩阵
    R = zyz_euler_to_rotation_matrix(rot_deg, tilt_deg, psi_deg)
    
    # 获取体积中心
    center = np.array(volume.shape) / 2.0 - 0.5
    
    # 创建坐标网格
    z, y, x = np.meshgrid(
        np.arange(volume.shape[0], dtype=float),
        np.arange(volume.shape[1], dtype=float),
        np.arange(volume.shape[2], dtype=float),
        indexing='ij'
    )
    
    # 将坐标转换为相对于中心的坐标
    coords = np.stack([z - center[0], y - center[1], x - center[2]], axis=-1)
    
    # 应用旋转：新坐标 = R^T * 旧坐标（因为我们旋转的是坐标系，不是体积）
    # 对于投影，我们需要将体积旋转，所以使用 R
    coords_rotated = np.dot(coords.reshape(-1, 3), R.T).reshape(coords.shape)
    
    # 将旋转后的坐标转换回索引
    z_rot = coords_rotated[:, :, :, 0] + center[0]
    y_rot = coords_rotated[:, :, :, 1] + center[1]
    x_rot = coords_rotated[:, :, :, 2] + center[2]
    
    # 使用插值获取旋转后的体积值
    from scipy.ndimage import map_coordinates
    rotated_volume = map_coordinates(
        volume,
        [z_rot, y_rot, x_rot],
        order=1,
        mode='constant',
        cval=0.0
    )
    
    # 处理旋转后边界区域的0值问题
    # 如果旋转后某些区域是0（可能是边界填充的0），用噪声替换以避免影响投影
    # 注意：这里只替换那些在原始体积边界外的0值，而不是所有0值
    # 简单方法：如果某个体素是0且周围有很多0，可能是边界区域，用噪声填充
    if mask is None:
        # 没有mask时，检测并填充边界0值区域
        # 计算噪声标准差（如果未提供）
        if noise_std is None:
            noise_std = np.std(volume)
        
        # 检测边界0值区域：如果整个Z列（沿投影方向）都是0，很可能是旋转后的边界区域
        # 这些0值是由map_coordinates的边界填充产生的，会影响投影结果
        zero_mask = (rotated_volume == 0.0)
        if np.any(zero_mask):
            # 检测哪些(Y, X)位置的整个Z列都是0（肯定是边界区域）
            z_all_zero = np.all(zero_mask, axis=0)  # shape: (Y, X)
            
            # 对于这些边界区域，用噪声填充以避免0值影响投影
            if np.any(z_all_zero):
                # 为边界区域生成噪声
                boundary_noise = np.random.normal(noise_mean, noise_std, size=rotated_volume.shape)
                # 只替换那些整个Z列都是0的位置
                for y_idx in range(rotated_volume.shape[1]):
                    for x_idx in range(rotated_volume.shape[2]):
                        if z_all_zero[y_idx, x_idx]:
                            rotated_volume[:, y_idx, x_idx] = boundary_noise[:, y_idx, x_idx]
    
    # 沿 Z 轴投影
    if mode == "sum":
        projection = np.sum(rotated_volume, axis=0)
    elif mode == "mean":
        projection = np.mean(rotated_volume, axis=0)
    elif mode == "max":
        projection = np.max(rotated_volume, axis=0)
    elif mode == "central_slice":
        center_z = rotated_volume.shape[0] // 2
        projection = rotated_volume[center_z, :, :]
    else:
        raise ValueError(f"Unknown projection mode: {mode}")
    
    # 重采样到目标尺寸
    if projection.shape != (box_size, box_size):
        from scipy.ndimage import zoom
        zoom_factors = (
            box_size / projection.shape[0],
            box_size / projection.shape[1]
        )
        projection = zoom(projection, zoom_factors, order=1)
    
    # 归一化
    if normalize:
        projection_min = projection.min()
        projection_max = projection.max()
        if projection_max > projection_min:
            projection = (projection - projection_min) / (projection_max - projection_min)
        else:
            projection = np.zeros_like(projection)
    
    return projection.astype(np.float32)


def project_3d_to_2d_rotated_gpu(
    volume: np.ndarray,
    rot_deg: float,
    tilt_deg: float,
    psi_deg: float,
    box_size: int,
    mode: Literal["sum", "mean", "max", "central_slice"] = "sum",
    normalize: bool = True,
    mask: np.ndarray = None,
    noise_mean: float = 0.0,
    noise_std: float = None,
    device: Optional[str] = None
) -> np.ndarray:
    """
    将 3D 体积按照给定的 ZYZ 欧拉角旋转后投影到 2D（GPU加速版本）。
    
    使用PyTorch的grid_sample进行3D旋转和插值，显著加速计算。
    
    Parameters
    ----------
    volume : np.ndarray
        3D 体积数组，形状为 (Z, Y, X)
    rot_deg : float
        Rot 角度（度）
    tilt_deg : float
        Tilt 角度（度）
    psi_deg : float
        Psi 角度（度）
    box_size : int
        输出图像尺寸（box_size × box_size）
    mode : str, optional
        投影模式，默认 "sum"。可选值：
        - "sum": 沿Z轴求和投影（累加所有层，适合增强信号）
        - "mean": 沿Z轴平均投影（取平均值，适合减少噪声）
        - "max": 沿Z轴最大投影（取每列最大值，适合突出最强信号）
        - "central_slice": 中心切片（取Z方向中心层，类似单层图像）
    normalize : bool, optional
        是否归一化输出，默认 True
    mask : np.ndarray, optional
        3D 二值 mask，形状与 volume 相同。True/1 表示保留原始信号，False/0 表示用噪声填充。
        如果为 None，则不应用 mask。
    noise_mean : float, optional
        Gaussian 噪声的均值，默认 0.0
    noise_std : float, optional
        Gaussian 噪声的标准差。如果为 None，则使用 volume 在 mask 外区域的标准差。
    device : str, optional
        PyTorch设备，如 'cuda:0' 或 'cpu'。如果为 None，则自动选择（优先使用GPU）。
        
    Returns
    -------
    np.ndarray
        2D 投影图像，形状为 (box_size, box_size)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available. Please install PyTorch to use GPU acceleration. "
            "You can use the CPU version project_3d_to_2d_rotated() instead."
        )
    
    # 验证输入 volume 的维度
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(
            f"volume must be 3D array, but got {volume.ndim}D array with shape {volume.shape}. "
            f"Expected shape: (Z, Y, X)"
        )
    
    # 选择设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 应用 mask（如果提供）
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != volume.shape:
            raise ValueError(f"mask shape {mask.shape} must match volume shape {volume.shape}")
        
        # 计算噪声标准差（如果未提供）
        if noise_std is None:
            # 使用 mask 外区域的标准差
            masked_out = volume[~mask]
            if len(masked_out) > 0:
                noise_std = np.std(masked_out)
            else:
                # 如果 mask 外没有数据，使用整个 volume 的标准差
                noise_std = np.std(volume)
        
        # 创建带噪声的 volume
        volume_masked = volume.copy()
        # 在 mask 外区域填充 Gaussian 噪声
        noise = np.random.normal(noise_mean, noise_std, size=volume.shape)
        volume_masked[~mask] = noise[~mask]
        volume = volume_masked
    
    # 获取旋转矩阵
    R = zyz_euler_to_rotation_matrix(rot_deg, tilt_deg, psi_deg)
    
    # 转换为PyTorch张量
    volume_tensor = torch.from_numpy(volume.astype(np.float32)).to(device)
    # PyTorch的grid_sample期望输入为 (N, C, D, H, W)，这里N=1, C=1
    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
    
    # 获取体积中心
    center = torch.tensor([volume.shape[0] / 2.0 - 0.5,
                          volume.shape[1] / 2.0 - 0.5,
                          volume.shape[2] / 2.0 - 0.5], dtype=torch.float32, device=device)
    
    # 创建坐标网格
    z, y, x = torch.meshgrid(
        torch.arange(volume.shape[0], dtype=torch.float32, device=device),
        torch.arange(volume.shape[1], dtype=torch.float32, device=device),
        torch.arange(volume.shape[2], dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 将坐标转换为相对于中心的坐标
    coords = torch.stack([
        z - center[0],
        y - center[1],
        x - center[2]
    ], dim=-1)  # (Z, Y, X, 3)
    
    # 应用旋转：新坐标 = R^T * 旧坐标
    R_tensor = torch.from_numpy(R.astype(np.float32)).to(device)
    coords_flat = coords.reshape(-1, 3)  # (Z*Y*X, 3)
    coords_rotated_flat = torch.matmul(coords_flat, R_tensor.T)  # (Z*Y*X, 3)
    coords_rotated = coords_rotated_flat.reshape(coords.shape)  # (Z, Y, X, 3)
    
    # 将旋转后的坐标转换回索引（grid_sample期望归一化坐标 [-1, 1]）
    # 归一化坐标：将物理坐标转换为 [-1, 1] 范围
    z_rot = coords_rotated[:, :, :, 0] + center[0]
    y_rot = coords_rotated[:, :, :, 1] + center[1]
    x_rot = coords_rotated[:, :, :, 2] + center[2]
    
    # 归一化到 [-1, 1] 范围
    z_norm = 2.0 * z_rot / (volume.shape[0] - 1) - 1.0
    y_norm = 2.0 * y_rot / (volume.shape[1] - 1) - 1.0
    x_norm = 2.0 * x_rot / (volume.shape[2] - 1) - 1.0
    
    # grid_sample期望的格式：(N, D, H, W, 3)，其中最后一维是 (x, y, z)
    # 注意：PyTorch的grid_sample使用 (x, y, z) 顺序，而我们的坐标是 (z, y, x)
    # 需要重新排列为 (x, y, z)
    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (Z, Y, X, 3) with (x, y, z) order
    grid = grid.unsqueeze(0)  # (1, Z, Y, X, 3)
    
    # 使用grid_sample进行3D旋转和插值
    rotated_volume = F.grid_sample(
        volume_tensor,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )
    
    # 移除batch和channel维度
    rotated_volume = rotated_volume.squeeze(0).squeeze(0)  # (Z, Y, X)
    
    # 处理旋转后边界区域的0值问题
    if mask is None:
        # 没有mask时，检测并填充边界0值区域
        # 计算噪声标准差（如果未提供）
        if noise_std is None:
            noise_std = float(torch.std(volume_tensor.squeeze()).cpu().numpy())
        
        # 检测边界0值区域：如果整个Z列都是0，很可能是旋转后的边界区域
        zero_mask = (rotated_volume == 0.0)
        if torch.any(zero_mask):
            # 检测哪些(Y, X)位置的整个Z列都是0
            z_all_zero = torch.all(zero_mask, dim=0)  # shape: (Y, X)
            
            # 对于这些边界区域，用噪声填充
            if torch.any(z_all_zero):
                boundary_noise = torch.randn(
                    rotated_volume.shape,
                    device=device,
                    dtype=torch.float32
                ) * noise_std + noise_mean
                # 只替换那些整个Z列都是0的位置
                for y_idx in range(rotated_volume.shape[1]):
                    for x_idx in range(rotated_volume.shape[2]):
                        if z_all_zero[y_idx, x_idx]:
                            rotated_volume[:, y_idx, x_idx] = boundary_noise[:, y_idx, x_idx]
    
    # 沿 Z 轴投影
    if mode == "sum":
        projection = torch.sum(rotated_volume, dim=0)
    elif mode == "mean":
        projection = torch.mean(rotated_volume, dim=0)
    elif mode == "max":
        projection = torch.max(rotated_volume, dim=0)[0]
    elif mode == "central_slice":
        center_z = rotated_volume.shape[0] // 2
        projection = rotated_volume[center_z, :, :]
    else:
        raise ValueError(f"Unknown projection mode: {mode}")
    
    # 重采样到目标尺寸
    if projection.shape != (box_size, box_size):
        # 使用PyTorch的interpolate进行重采样
        projection = projection.unsqueeze(0).unsqueeze(0)  # (1, 1, Y, X)
        projection = F.interpolate(
            projection,
            size=(box_size, box_size),
            mode='bilinear',
            align_corners=True
        )
        projection = projection.squeeze(0).squeeze(0)  # (box_size, box_size)
    
    # 归一化
    if normalize:
        projection_min = torch.min(projection)
        projection_max = torch.max(projection)
        if projection_max > projection_min:
            projection = (projection - projection_min) / (projection_max - projection_min)
        else:
            projection = torch.zeros_like(projection)
    
    # 转换回numpy数组
    return projection.cpu().numpy().astype(np.float32)

