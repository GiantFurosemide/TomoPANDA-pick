"""
3D 到 2D 投影模块

提供将 3D 体积投影到 2D 图像的功能，支持多种投影模式。
"""

import numpy as np
from typing import Literal
import os
import importlib.util

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
        投影模式，默认 "sum"
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

