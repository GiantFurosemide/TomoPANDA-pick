#!/usr/bin/env python
"""
基于球面 Fibonacci 均匀方向采样的 subtomo 多取向投影生成主脚本

使用：python main.py -i config.yaml
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import mrcfile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 导入项目模块
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from utils.io_starfile import create_relion3_star

# 导入本地模块（使用 importlib 处理数字开头的模块名）
import importlib.util
math_utils_path = os.path.join(os.path.dirname(__file__), 'math_utils.py')
spec = importlib.util.spec_from_file_location("math_utils", math_utils_path)
math_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math_utils)
fibonacci_sphere_sampling = math_utils.fibonacci_sphere_sampling
direction_to_relion_zyz = math_utils.direction_to_relion_zyz

projection_path = os.path.join(os.path.dirname(__file__), 'projection.py')
spec = importlib.util.spec_from_file_location("projection", projection_path)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)
project_3d_to_2d_rotated = projection.project_3d_to_2d_rotated


def read_subtomo_list(txt_path: str) -> pd.DataFrame:
    """
    读取 subtomo 列表文件（每行一个 MRC 路径）。
    
    Parameters
    ----------
    txt_path : str
        TXT 文件路径
        
    Returns
    -------
    pd.DataFrame
        包含 'subtomo_id' 和 'subtomo_path' 列的 DataFrame
    """
    subtomo_paths = []
    with open(txt_path, 'r') as f:
        for line in f:
            path = line.strip()
            if path and not path.startswith('#'):
                subtomo_paths.append(path)
    
    df = pd.DataFrame({
        'subtomo_id': range(len(subtomo_paths)),
        'subtomo_path': subtomo_paths
    })
    
    return df


def generate_orientations(n_orientations: int, psi_deg: float = 0.0) -> pd.DataFrame:
    """
    生成 N 个 Fibonacci 球面方向并转换为 ZYZ 欧拉角。
    
    Parameters
    ----------
    n_orientations : int
        方向数量
    psi_deg : float, optional
        Psi 角度（度），默认 0.0
        
    Returns
    -------
    pd.DataFrame
        包含方向信息的 DataFrame
    """
    # 生成 Fibonacci 球面方向
    directions = fibonacci_sphere_sampling(n_orientations)
    
    # 转换为 ZYZ 欧拉角
    angles = direction_to_relion_zyz(directions, psi_deg=psi_deg)
    
    # 创建 DataFrame
    df = pd.DataFrame({
        'orientation_id': range(n_orientations),
        'vx': directions[:, 0],
        'vy': directions[:, 1],
        'vz': directions[:, 2],
        'rot_deg': angles[:, 0],
        'tilt_deg': angles[:, 1],
        'psi_deg': angles[:, 2]
    })
    
    return df


def create_global_particle_plan(subtomo_df: pd.DataFrame, orientations_df: pd.DataFrame, 
                                 output_root: str, orientation_dir_pattern: str,
                                 stack_name_pattern: str) -> pd.DataFrame:
    """
    创建全局粒子计划表。
    
    Parameters
    ----------
    subtomo_df : pd.DataFrame
        Subtomo 列表
    orientations_df : pd.DataFrame
        方向列表
    output_root : str
        输出根目录
    orientation_dir_pattern : str
        方向目录模式，如 "ori_{k:03d}"
    stack_name_pattern : str
        栈文件名模式，如 "proj_ori_{k:03d}.mrcs"
        
    Returns
    -------
    pd.DataFrame
        全局粒子计划表
    """
    records = []
    particle_global_id = 0
    
    for _, subtomo_row in subtomo_df.iterrows():
        subtomo_id = subtomo_row['subtomo_id']
        subtomo_path = subtomo_row['subtomo_path']
        
        for _, ori_row in orientations_df.iterrows():
            orientation_id = int(ori_row['orientation_id'])
            rot_deg = ori_row['rot_deg']
            tilt_deg = ori_row['tilt_deg']
            psi_deg = ori_row['psi_deg']
            vx = ori_row['vx']
            vy = ori_row['vy']
            vz = ori_row['vz']
            
            # 生成相对路径
            ori_dir = orientation_dir_pattern.format(k=orientation_id)
            stack_name = stack_name_pattern.format(k=orientation_id)
            stack_relpath = os.path.join(ori_dir, stack_name)
            
            # slice_index 是 subtomo_id 的顺序（1-based）
            slice_index = subtomo_id + 1
            
            records.append({
                'particle_global_id': particle_global_id,
                'orientation_id': orientation_id,
                'subtomo_id': subtomo_id,
                'subtomo_path': subtomo_path,
                'stack_relpath': stack_relpath,
                'slice_index_in_stack': slice_index,
                'rot_deg': rot_deg,
                'tilt_deg': tilt_deg,
                'psi_deg': psi_deg,
                'vx': vx,
                'vy': vy,
                'vz': vz
            })
            
            particle_global_id += 1
    
    return pd.DataFrame(records)


def generate_projection_stacks(global_plan_df: pd.DataFrame, subtomo_df: pd.DataFrame,
                               orientations_df: pd.DataFrame, output_root: str,
                               box_size: int, mode: str, normalize: bool,
                               orientation_dir_pattern: str, stack_name_pattern: str,
                               mask_path: str = None, noise_mean: float = 0.0, noise_std: float = None,
                               num_threads: int = None):
    """
    按方向生成投影栈。
    
    Parameters
    ----------
    global_plan_df : pd.DataFrame
        全局粒子计划表
    subtomo_df : pd.DataFrame
        Subtomo 列表
    orientations_df : pd.DataFrame
        方向列表
    output_root : str
        输出根目录
    box_size : int
        输出图像尺寸
    mode : str
        投影模式
    normalize : bool
        是否归一化
    orientation_dir_pattern : str
        方向目录模式
    stack_name_pattern : str
        栈文件名模式
    mask_path : str, optional
        Mask 文件路径（MRC 格式）。如果提供，mask 内的区域保留原始信号，mask 外用噪声填充。
        如果为 None，则不应用 mask。
    noise_mean : float, optional
        Gaussian 噪声的均值，默认 0.0
    noise_std : float, optional
        Gaussian 噪声的标准差。如果为 None，则自动计算。
    num_threads : int, optional
        线程数。如果为 None，则使用 CPU 核心数。
    """
    n_orientations = len(orientations_df)
    n_subtomos = len(subtomo_df)
    
    # 设置线程数
    if num_threads is None:
        num_threads = os.cpu_count() or 1
    
    # 加载 mask（如果提供）
    mask = None
    if mask_path is not None and os.path.exists(mask_path):
        with mrcfile.open(mask_path) as mrc:
            mask = mrc.data.astype(bool)
        print(f"Loaded mask from {mask_path}, shape: {mask.shape}")
    
    # 按方向分组处理
    for orientation_id in tqdm(range(n_orientations), desc="Generating projection stacks"):
        ori_row = orientations_df.iloc[orientation_id]
        rot_deg = ori_row['rot_deg']
        tilt_deg = ori_row['tilt_deg']
        psi_deg = ori_row['psi_deg']
        
        # 创建方向目录
        ori_dir = orientation_dir_pattern.format(k=orientation_id)
        ori_dir_path = os.path.join(output_root, ori_dir)
        os.makedirs(ori_dir_path, exist_ok=True)
        
        # 创建栈文件
        stack_name = stack_name_pattern.format(k=orientation_id)
        stack_path = os.path.join(ori_dir_path, stack_name)
        
        # 获取该方向的所有 subtomo
        ori_plan = global_plan_df[global_plan_df['orientation_id'] == orientation_id]
        ori_plan = ori_plan.sort_values('subtomo_id')
        
        # 创建 MRC 栈文件（逐片写入，避免一次性加载所有数据到内存）
        # 注意：MRC 格式要求预先知道文件大小，初始化时需要临时内存
        # 但之后使用内存映射方式逐片写入，处理过程中不会占用大量内存
        stack_shape = (n_subtomos, box_size, box_size)
        
        # 步骤1：创建文件并初始化 header（需要临时内存，但立即释放）
        with mrcfile.new(stack_path, overwrite=True) as mrc_stack:
            init_data = np.zeros(stack_shape, dtype=np.float32)
            mrc_stack.set_data(init_data)
            mrc_stack.update_header_from_data()
            del init_data  # 立即释放临时内存
        
        # 步骤2：使用内存映射方式打开文件，多线程并行处理 subtomo
        # 定义处理单个 subtomo 的函数
        def process_subtomo(plan_row):
            """处理单个 subtomo 并返回结果"""
            subtomo_path = plan_row['subtomo_path']
            slice_idx = int(plan_row['slice_index_in_stack']) - 1
            
            # 读取 subtomo 并生成投影
            with mrcfile.open(subtomo_path) as mrc:
                volume = mrc.data
                
                # 获取 MRC header 中的实际维度（可能被压缩）
                header_shape = (mrc.header.nz, mrc.header.ny, mrc.header.nx)
                
                # 如果 volume 维度被压缩（比如某个维度是1），尝试恢复
                if volume.ndim == 2 and len(header_shape) == 3:
                    # 检查是否是单层的情况（Z=1）
                    if header_shape[0] == 1:
                        volume = volume[np.newaxis, :, :]  # 添加 Z 维度
                    elif header_shape[1] == 1:
                        volume = volume[:, np.newaxis, :]  # 添加 Y 维度
                    elif header_shape[2] == 1:
                        volume = volume[:, :, np.newaxis]  # 添加 X 维度
                    else:
                        # 无法确定如何恢复，报错
                        raise ValueError(
                            f"Subtomo file '{subtomo_path}' has inconsistent dimensions: "
                            f"header shape {header_shape} but data shape {volume.shape}. "
                            f"This might indicate a corrupted or incorrectly formatted MRC file."
                        )
                elif volume.ndim == 1:
                    # 1D 数组，尝试根据 header 重塑
                    if len(header_shape) == 3:
                        volume = volume.reshape(header_shape)
                    else:
                        raise ValueError(
                            f"Subtomo file '{subtomo_path}' contains 1D data with shape {volume.shape}, "
                            f"but 3D volume is required. Header shape: {header_shape}"
                        )
                
                # 验证 volume 是 3D 数组
                if volume.ndim != 3:
                    raise ValueError(
                        f"Subtomo file '{subtomo_path}' contains {volume.ndim}D data with shape {volume.shape} "
                        f"(header indicates {header_shape}), but 3D volume is required (expected shape: (Z, Y, X)). "
                        f"Please check the MRC file format. "
                        f"File might be corrupted or in an unexpected format."
                    )
                
                # 检查 mask 形状是否匹配
                if mask is not None and mask.shape != volume.shape:
                    current_mask = None
                else:
                    current_mask = mask
            
            # 生成投影
            proj = project_3d_to_2d_rotated(
                volume, rot_deg, tilt_deg, psi_deg,
                box_size, mode=mode, normalize=normalize,
                mask=current_mask, noise_mean=noise_mean, noise_std=noise_std
            )
            
            return slice_idx, proj
        
        # 使用线程池并行处理
        with mrcfile.open(stack_path, mode='r+', permissive=True) as mrc_stack:
            # 使用锁来保护写入和 flush 操作（虽然每个线程写入不同的 slice，但为了安全起见使用锁）
            write_lock = threading.Lock()
            processed_count = [0]  # 使用列表以便在线程间共享
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # 提交所有任务
                future_to_row = {
                    executor.submit(process_subtomo, plan_row): plan_row 
                    for _, plan_row in ori_plan.iterrows()
                }
                
                # 处理完成的任务
                for future in tqdm(as_completed(future_to_row), 
                                  total=len(future_to_row), 
                                  desc=f"Processing orientation {orientation_id}",
                                  leave=False):
                    try:
                        slice_idx, proj = future.result()
                        # 写入到栈文件的对应位置（使用锁保护，确保线程安全）
                        with write_lock:
                            mrc_stack.data[slice_idx] = proj
                            processed_count[0] += 1
                            # 定期刷新到磁盘（每100个刷新一次，平衡性能和安全性）
                            if processed_count[0] % 100 == 0:
                                mrc_stack.flush()
                    except Exception as e:
                        plan_row = future_to_row[future]
                        subtomo_path = plan_row.get('subtomo_path', 'unknown')
                        subtomo_id = plan_row.get('subtomo_id', 'unknown')
                        print(f"\n{'='*80}")
                        print(f"ERROR processing subtomo:")
                        print(f"  Subtomo ID: {subtomo_id}")
                        print(f"  File path: {subtomo_path}")
                        print(f"  Error type: {type(e).__name__}")
                        print(f"  Error message: {str(e)}")
                        print(f"{'='*80}\n")
                        # 重新抛出异常以便用户知道有问题
                        raise
                        raise
            
            # 最后更新 header 并确保所有数据写入磁盘
            mrc_stack.update_header_from_data()
            mrc_stack.flush()


def write_per_orientation_index(global_plan_df: pd.DataFrame, output_root: str,
                                per_orientation_index_pattern: str):
    """
    写入每个方向的索引文件。
    
    Parameters
    ----------
    global_plan_df : pd.DataFrame
        全局粒子计划表
    output_root : str
        输出根目录
    per_orientation_index_pattern : str
        每个方向索引文件模式
    """
    n_orientations = global_plan_df['orientation_id'].max() + 1
    
    for orientation_id in range(n_orientations):
        ori_plan = global_plan_df[global_plan_df['orientation_id'] == orientation_id]
        ori_plan = ori_plan.sort_values('subtomo_id')
        
        # 创建索引 DataFrame
        index_df = pd.DataFrame({
            'slice_index': ori_plan['slice_index_in_stack'],
            'subtomo_id': ori_plan['subtomo_id'],
            'subtomo_path': ori_plan['subtomo_path'],
            'orientation_id': ori_plan['orientation_id'],
            'rot_deg': ori_plan['rot_deg'],
            'tilt_deg': ori_plan['tilt_deg'],
            'psi_deg': ori_plan['psi_deg']
        })
        
        # 写入文件
        index_filename = per_orientation_index_pattern.format(k=orientation_id)
        index_path = os.path.join(output_root, index_filename)
        index_df.to_csv(index_path, sep='\t', index=False)


def write_global_index(global_plan_df: pd.DataFrame, output_root: str,
                      global_index_table: str):
    """
    写入全局索引文件。
    
    Parameters
    ----------
    global_plan_df : pd.DataFrame
        全局粒子计划表
    output_root : str
        输出根目录
    global_index_table : str
        全局索引文件名
    """
    index_df = global_plan_df[[
        'particle_global_id',
        'orientation_id',
        'subtomo_id',
        'subtomo_path',
        'stack_relpath',
        'slice_index_in_stack',
        'rot_deg',
        'tilt_deg',
        'psi_deg',
        'vx',
        'vy',
        'vz'
    ]]
    
    index_path = os.path.join(output_root, global_index_table)
    index_df.to_csv(index_path, sep='\t', index=False)


def write_relion_star(global_plan_df: pd.DataFrame, output_root: str,
                     relion_star: str, optics_config: dict, box_size: int):
    """
    写入 RELION 3 格式的 particle.star 文件。
    
    Parameters
    ----------
    global_plan_df : pd.DataFrame
        全局粒子计划表
    output_root : str
        输出根目录
    relion_star : str
        RELION star 文件名
    optics_config : dict
        Optics 配置字典
    box_size : int
        图像尺寸（box_size），用于添加到 optics 部分
    """
    # 准备图像名称（相对于 star 文件所在目录）
    star_dir = os.path.dirname(os.path.join(output_root, relion_star))
    if star_dir == '':
        star_dir = '.'
    
    image_names = []
    angles_list = []
    
    for _, row in global_plan_df.iterrows():
        stack_relpath = row['stack_relpath']
        slice_index = row['slice_index_in_stack']
        
        # 图像名称格式：slice_index@stack_relpath
        image_name = f"{slice_index}@{stack_relpath}"
        image_names.append(image_name)
        
        angles_list.append([
            row['rot_deg'],
            row['tilt_deg'],
            row['psi_deg']
        ])
    
    angles = np.array(angles_list)
    
    # 创建 RELION 3 star 文件
    star_path = os.path.join(output_root, relion_star)
    create_relion3_star(
        image_names=image_names,
        angles=angles,
        optics_group_id=optics_config['optics_group_id'],
        optics_group_name=optics_config['optics_group_name'],
        pixel_size=optics_config['pixel_size'],
        voltage_kv=optics_config['voltage_kv'],
        cs_mm=optics_config['cs_mm'],
        amplitude_contrast=optics_config['amplitude_contrast'],
        output_file=star_path,
        image_size=box_size
    )


def write_per_orientation_star(global_plan_df: pd.DataFrame, output_root: str,
                               star_pattern: str, optics_config: dict, box_size: int):
    """
    为每个 orientation 写入单独的 RELION 3 格式的 particle.star 文件。
    
    Parameters
    ----------
    global_plan_df : pd.DataFrame
        全局粒子计划表
    output_root : str
        输出根目录
    star_pattern : str
        Star 文件名模式，如 "particles_ori_{k:03d}.star"
    optics_config : dict
        Optics 配置字典
    box_size : int
        图像尺寸（box_size），用于添加到 optics 部分
    """
    n_orientations = int(global_plan_df['orientation_id'].max() + 1)
    
    for orientation_id in tqdm(range(n_orientations), desc="Writing per-orientation star files"):
        # 筛选该 orientation 的所有 particles
        ori_plan = global_plan_df[global_plan_df['orientation_id'] == orientation_id]
        ori_plan = ori_plan.sort_values('subtomo_id')
        
        if len(ori_plan) == 0:
            continue
        
        # 准备图像名称和角度
        image_names = []
        angles_list = []
        
        for _, row in ori_plan.iterrows():
            stack_relpath = row['stack_relpath']
            slice_index = row['slice_index_in_stack']
            
            # 图像名称格式：slice_index@stack_relpath
            image_name = f"{slice_index}@{stack_relpath}"
            image_names.append(image_name)
            
            angles_list.append([
                row['rot_deg'],
                row['tilt_deg'],
                row['psi_deg']
            ])
        
        angles = np.array(angles_list)
        
        # 生成 star 文件名
        star_filename = star_pattern.format(k=orientation_id)
        star_path = os.path.join(output_root, star_filename)
        
        # 创建 RELION 3 star 文件
        create_relion3_star(
            image_names=image_names,
            angles=angles,
            optics_group_id=optics_config['optics_group_id'],
            optics_group_name=optics_config['optics_group_name'],
            pixel_size=optics_config['pixel_size'],
            voltage_kv=optics_config['voltage_kv'],
            cs_mm=optics_config['cs_mm'],
            amplitude_contrast=optics_config['amplitude_contrast'],
            output_file=star_path,
            image_size=box_size
        )


def main():
    parser = argparse.ArgumentParser(
        description='Generate 2D projections from subtomos using Fibonacci sphere sampling'
    )
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Path to YAML configuration file')
    args = parser.parse_args()
    
    # 读取配置
    with open(args.input, 'r') as f:
        config = yaml.safe_load(f)
    
    # 解析配置
    subtomo_txt = config['input']['subtomo_txt']
    n_orientations = config['orientations']['N_orientations']
    psi_deg = config['orientations']['psi_deg']
    output_root = config['output']['root_dir']
    orientation_dir_pattern = config['output']['orientation_dir_pattern']
    stack_name_pattern = config['output']['stack_name_pattern']
    orientations_table = config['output']['orientations_table']
    global_index_table = config['output']['global_index_table']
    per_orientation_index_pattern = config['output']['per_orientation_index_pattern']
    relion_star = config['output']['relion_star']
    box_size = config['projection']['box_size']
    mode = config['projection']['mode']
    normalize = config['projection']['normalize']
    optics_config = config['optics']
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 1. 读取 subtomo 列表
    print("Reading subtomo list...")
    subtomo_df = read_subtomo_list(subtomo_txt)
    print(f"Found {len(subtomo_df)} subtomos")
    
    # 2. 生成方向
    print(f"Generating {n_orientations} orientations...")
    orientations_df = generate_orientations(n_orientations, psi_deg)
    
    # 保存方向表
    orientations_path = os.path.join(output_root, orientations_table)
    orientations_df.to_csv(orientations_path, sep='\t', index=False)
    print(f"Saved orientations to {orientations_path}")
    
    # 3. 创建全局粒子计划
    print("Creating global particle plan...")
    global_plan_df = create_global_particle_plan(
        subtomo_df, orientations_df, output_root,
        orientation_dir_pattern, stack_name_pattern
    )
    print(f"Created plan for {len(global_plan_df)} particles")
    
    # 4. 生成投影栈
    print("Generating projection stacks...")
    # 读取 mask 相关配置（可选）
    mask_path = config['projection'].get('mask_path', None)
    noise_mean = config['projection'].get('noise_mean', 0.0)
    noise_std = config['projection'].get('noise_std', None)
    # 读取线程数配置（可选）
    num_threads = config.get('processing', {}).get('num_threads', None)
    
    generate_projection_stacks(
        global_plan_df, subtomo_df, orientations_df, output_root,
        box_size, mode, normalize, orientation_dir_pattern, stack_name_pattern,
        mask_path=mask_path, noise_mean=noise_mean, noise_std=noise_std,
        num_threads=num_threads
    )
    
    # 5. 写入 per-orientation index
    print("Writing per-orientation indices...")
    write_per_orientation_index(
        global_plan_df, output_root, per_orientation_index_pattern
    )
    
    # 6. 写入 global index
    print("Writing global index...")
    write_global_index(global_plan_df, output_root, global_index_table)
    
    # 7. 写入总的 RELION star 文件（包含所有 orientations）
    print("Writing RELION star file...")
    write_relion_star(global_plan_df, output_root, relion_star, optics_config, box_size)
    print(f"Saved RELION star file to {os.path.join(output_root, relion_star)}")
    
    # 8. 为每个 orientation 写入单独的 RELION star 文件
    print("Writing per-orientation RELION star files...")
    # 从 relion_star 文件名生成模式
    # 例如 "particles.star" -> "particles_ori_{k:03d}.star"
    star_basename = os.path.splitext(relion_star)[0]  # 去掉扩展名
    star_pattern = f"{star_basename}_ori_{{k:03d}}.star"
    write_per_orientation_star(global_plan_df, output_root, star_pattern, optics_config, box_size)
    print(f"Saved per-orientation RELION star files to {output_root}")
    
    # 9. 保存解析后的配置
    config_resolved_path = os.path.join(output_root, 'config_resolved.yaml')
    with open(config_resolved_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved resolved config to {config_resolved_path}")
    
    print("Done!")


if __name__ == '__main__':
    main()

