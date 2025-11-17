"""
数学工具模块：球面 Fibonacci 采样和欧拉角转换

提供：
1. Fibonacci 球面均匀方向采样
2. 方向向量到 RELION ZYZ 欧拉角转换
"""

import numpy as np


def fibonacci_sphere_sampling(n: int) -> np.ndarray:
    """
    生成 N 个在单位球面上均匀分布的 Fibonacci 采样点。
    
    对 k = 0 ... N-1：
    - t_k = (k + 0.5) / N
    - z_k = 1 - 2*t_k
    - r_k = sqrt(1 - z_k^2)
    - phi_k = 2*pi * k / golden_ratio
    - x_k = r_k * cos(phi_k)
    - y_k = r_k * sin(phi_k)
    
    返回方向向量 (vx, vy, vz)，形状为 (N, 3)。
    
    Parameters
    ----------
    n : int
        采样点数量
        
    Returns
    -------
    np.ndarray
        形状为 (N, 3) 的数组，每行是一个单位方向向量 (vx, vy, vz)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    # 黄金比例
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    # 生成索引
    k = np.arange(n, dtype=float)
    
    # 计算 z 坐标
    t = (k + 0.5) / n
    z = 1 - 2 * t
    
    # 计算半径（在 xy 平面上的投影半径）
    r = np.sqrt(1 - z * z)
    
    # 计算方位角
    phi = 2 * np.pi * k / golden_ratio
    
    # 计算 x, y 坐标
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    # 组合成方向向量
    directions = np.stack([x, y, z], axis=1)
    
    return directions


def direction_to_relion_zyz(direction: np.ndarray, psi_deg: float = 0.0) -> np.ndarray:
    """
    将方向向量转换为 RELION ZYZ 欧拉角。
    
    RELION ZYZ 约定：
    R = R_Z(Psi) * R_Y(Tilt) * R_Z(Rot)
    
    目标：旋转后把 Z 轴对准方向向量 v。
    
    转换公式：
    1. Tilt = arccos(vz)  [绕 Y 轴旋转]
    2. Rot = atan2(vy, vx)  [绕 Z 轴旋转]
    3. Psi = psi_deg  [配置给定，通常=0]
    
    Parameters
    ----------
    direction : np.ndarray
        方向向量，形状为 (3,) 或 (N, 3)
    psi_deg : float, optional
        Psi 角度（度），默认 0.0
        
    Returns
    -------
    np.ndarray
        ZYZ 欧拉角 (Rot, Tilt, Psi)，形状与 direction 相同（但最后一维为 3）
        单位为度
    """
    direction = np.asarray(direction, dtype=float)
    single = False
    
    if direction.ndim == 1:
        direction = direction.reshape(1, 3)
        single = True
    
    # 归一化方向向量（确保是单位向量）
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = direction / norms
    
    # 提取分量
    vx = direction[:, 0]
    vy = direction[:, 1]
    vz = direction[:, 2]
    
    # 计算 Tilt（绕 Y 轴）
    # 注意：arccos 的范围是 [0, pi]，这正好对应 Tilt 的范围
    tilt_deg = np.arccos(np.clip(vz, -1.0, 1.0)) * 180.0 / np.pi
    
    # 计算 Rot（绕 Z 轴）
    rot_deg = np.arctan2(vy, vx) * 180.0 / np.pi
    
    # Psi 固定为配置值
    psi_deg_arr = np.full(len(direction), psi_deg)
    
    # 组合成欧拉角
    angles = np.stack([rot_deg, tilt_deg, psi_deg_arr], axis=1)
    
    if single:
        return angles.reshape(3,)
    return angles


def zyz_euler_to_rotation_matrix(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    """
    将 RELION ZYZ 欧拉角转换为旋转矩阵。
    
    R = R_Z(Psi) * R_Y(Tilt) * R_Z(Rot)
    
    Parameters
    ----------
    rot_deg : float
        Rot 角度（度）
    tilt_deg : float
        Tilt 角度（度）
    psi_deg : float
        Psi 角度（度）
        
    Returns
    -------
    np.ndarray
        3x3 旋转矩阵
    """
    # 转换为弧度
    rot = np.deg2rad(rot_deg)
    tilt = np.deg2rad(tilt_deg)
    psi = np.deg2rad(psi_deg)
    
    # 绕 Z 轴旋转 Rot
    cos_rot = np.cos(rot)
    sin_rot = np.sin(rot)
    R_z1 = np.array([
        [cos_rot, -sin_rot, 0],
        [sin_rot,  cos_rot, 0],
        [0,        0,       1]
    ])
    
    # 绕 Y 轴旋转 Tilt
    cos_tilt = np.cos(tilt)
    sin_tilt = np.sin(tilt)
    R_y = np.array([
        [cos_tilt,  0, sin_tilt],
        [0,         1, 0],
        [-sin_tilt, 0, cos_tilt]
    ])
    
    # 绕 Z 轴旋转 Psi
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    R_z2 = np.array([
        [cos_psi, -sin_psi, 0],
        [sin_psi,  cos_psi, 0],
        [0,        0,       1]
    ])
    
    # 组合：R = R_Z(Psi) * R_Y(Tilt) * R_Z(Rot)
    R = R_z2 @ R_y @ R_z1
    
    return R

