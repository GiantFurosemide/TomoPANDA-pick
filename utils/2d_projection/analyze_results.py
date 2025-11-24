#!/usr/bin/env python
"""
2D投影结果分析脚本

提供从particle star文件中提取slice index，以及从txt文件中提取指定行的功能。

主要功能：
1. 从RELION particle star文件中提取所有的slice index（n值）
   - star文件中的_rlnImageName格式为：n@x.mrcs，其中n是slice index（从1开始）
   - 可以将0-based的indices保存到txt文件（每行一个index，从0开始）

2. 从txt文件中提取指定index对应的行
   - 支持0-based或1-based的indices
   - 可以保存提取的行到新的txt文件

3. 整合功能：从star文件提取indices，然后从txt文件提取对应的行
   - 输入：star文件和txt文件
   - 输出：提取的行保存到txt文件
   - 可选：同时保存0-based indices到单独的txt文件

4. 从颗粒路径中提取颗粒ID
   - 从路径basename中提取颗粒ID（如particle_035948.mrc -> 35948）
   - 支持从txt文件中批量提取所有颗粒ID

5. 从Dynamo tbl文件中提取指定颗粒ID对应的行
   - tbl文件的第一列（tag列）是颗粒ID
   - 可以匹配并提取所有与给定颗粒ID相同的行

6. 整合功能：从txt文件提取颗粒ID，然后从tbl文件提取对应的行
   - 输入：txt文件（包含颗粒路径）和tbl文件
   - 输出：提取的行保存到新的tbl文件

7. 完整流程整合功能：从star文件提取indices，从particle txt提取对应行，提取颗粒ID，从tbl提取对应行
   - 输入：star文件、particle txt文件、tbl文件、输出目录
   - 输出：在输出目录中生成index.txt、particle txt处理文件、tbl处理文件
   - 输出文件命名：原文件名前加".processed."，如 particles.txt -> particles.processed.txt

使用方法：

Python API:
    >>> from utils.2d_projection.analyze_results import (
    ...     extract_slice_indices_from_star,
    ...     extract_lines_by_indices,
    ...     extract_lines_from_star_and_txt
    ... )
    >>> 
    >>> # 方法1：只提取indices并保存到文件
    >>> indices = extract_slice_indices_from_star("particles.star", "indices.txt")
    >>> # indices.txt 内容：每行一个0-based的index
    >>> 
    >>> # 方法2：从txt文件提取指定行（使用1-based indices）
    >>> lines = extract_lines_by_indices("subtomos.txt", [1, 3, 5], "output.txt")
    >>> 
    >>> # 方法3：从txt文件提取指定行（使用0-based indices）
    >>> lines = extract_lines_by_indices("subtomos.txt", [0, 2, 4], "output.txt", indices_are_0based=True)
    >>> 
    >>> # 方法4：整合功能 - 从star提取indices，然后从txt提取对应行
    >>> lines = extract_lines_from_star_and_txt(
    ...     "particles.star",
    ...     "subtomos.txt", 
    ...     "extracted_lines.txt",
    ...     "indices.txt"  # 可选：保存indices
    ... )
    >>> 
    >>> # 方法5：从txt文件提取颗粒ID
    >>> from utils.2d_projection.analyze_results import (
    ...     extract_particle_ids_from_txt,
    ...     filter_dynamo_tbl_by_particle_ids,
    ...     extract_tbl_by_particle_txt
    ... )
    >>> particle_ids = extract_particle_ids_from_txt("particles.txt")
    >>> 
    >>> # 方法6：从tbl文件提取指定颗粒ID的行
    >>> filtered_df = filter_dynamo_tbl_by_particle_ids(
    ...     "all_particles.tbl",
    ...     [35948, 1234, 5678],
    ...     "filtered_particles.tbl"
    ... )
    >>> 
    >>> # 方法7：整合功能 - 从txt提取颗粒ID，然后从tbl提取对应行
    >>> filtered_df = extract_tbl_by_particle_txt(
    ...     "particles.txt",
    ...     "all_particles.tbl",
    ...     "filtered_particles.tbl"
    ... )

命令行使用:
    # 从star文件提取indices并保存（0-based）
    python utils/2d_projection/analyze_results.py -s particles.star --index-file indices.txt
    
    # 整合功能：从star和txt提取行
    python utils/2d_projection/analyze_results.py -s particles.star -t subtomos.txt -o output.txt --index-file indices.txt
    
    # 从txt文件提取指定行（使用1-based indices）
    python utils/2d_projection/analyze_results.py -t subtomos.txt -i 1 3 5 -o output.txt
    
    # 从txt文件提取指定行（使用0-based indices）
    python utils/2d_projection/analyze_results.py -t subtomos.txt -i 0 2 4 -o output.txt --zero-based
    
    # 从txt文件提取颗粒ID，然后从tbl文件提取对应的行
    python utils/2d_projection/analyze_results.py --extract-tbl-by-txt -t particles.txt --tbl all_particles.tbl --output-tbl filtered_particles.tbl
    
    # 完整流程：从star提取indices，从particle txt提取行，提取颗粒ID，从tbl提取行
    python utils/2d_projection/analyze_results.py --process-star-txt-tbl -s particles.star -t particles.txt --tbl all_particles.tbl --output-dir output_dir

注意事项：
- star文件中的slice index（n@x.mrcs中的n）是从1开始的（1-based）
- 保存到txt文件的indices会自动转换为0-based（n-1）
- 从txt文件读取行时，默认使用1-based indices（与star文件对应）
- 如果使用--zero-based参数，则使用0-based indices
- 颗粒路径格式应为：particle_035948.mrc（particle_前缀 + 数字ID + 扩展名）
- 颗粒ID会自动去掉前导零（如035948 -> 35948）
- tbl文件的第一列（tag列）必须是颗粒ID
"""

import starfile
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Set, List, Union

# 导入io_dynamo模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
from utils.io_dynamo import read_dynamo_tbl, COLUMNS_NAME


def extract_slice_indices_from_star(
    star_file: Union[str, Path],
    output_index_file: Union[str, Path] = None
) -> Set[int]:
    """
    从RELION particle star文件中提取所有的slice index（n值）。
    
    star文件中的_rlnImageName格式为：n@x.mrcs，其中n是slice index（从1开始）。
    
    Parameters
    ----------
    star_file : str or Path
        RELION particle star文件路径
    output_index_file : str or Path, optional
        如果提供，将0-based的indices保存到此txt文件（每行一个index，从0开始）
        
    Returns
    -------
    Set[int]
        所有唯一的slice index的集合（n值，1-based）
        
    Examples
    --------
    >>> indices = extract_slice_indices_from_star("particles.star", "indices.txt")
    >>> print(indices)
    {1, 2, 3, 5, 10, ...}
    >>> # indices.txt内容：0, 1, 2, 4, 9, ... (0-based)
    """
    star_file = Path(star_file)
    if not star_file.exists():
        raise FileNotFoundError(f"Star file not found: {star_file}")
    
    # 读取star文件（使用always_dict=False，参考io_dynamo.py的实现）
    star_data = starfile.read(star_file, always_dict=False)
    
    # 如果star文件包含多个block（如optics和particles），取particles block
    particles_df = None
    if isinstance(star_data, dict):
        # If dict, look for particles and optics blocks
        if 'particles' in star_data:
            particles_df = star_data['particles']
        elif len(star_data) > 0:
            # Get first DataFrame value as particles
            df_candidates = [v for v in star_data.values() if isinstance(v, pd.DataFrame)]
            if len(df_candidates) > 0:
                particles_df = df_candidates[0]
            else:
                raise ValueError(f"Could not find DataFrame in STAR file dict: {star_file}")
        else:
            raise ValueError(f"Could not find particle data block in STAR file: {star_file}")
    elif isinstance(star_data, pd.DataFrame):
        particles_df = star_data
    else:
        raise ValueError(f"Unexpected data type from starfile.read: {type(star_data)}")
    
    # 检查列名（starfile可能返回带或不带下划线的列名）
    # 尝试两种可能的列名格式：rlnImageName 或 _rlnImageName
    image_name_col = None
    for col_name in ['rlnImageName', '_rlnImageName']:
        if col_name in particles_df.columns:
            image_name_col = col_name
            break
    
    if image_name_col is None:
        # 打印所有可用的列名以便调试
        available_cols = list(particles_df.columns)
        raise ValueError(
            f"Star file does not contain 'rlnImageName' or '_rlnImageName' column. "
            f"Available columns: {available_cols}"
        )
    
    # 提取所有的slice index（1-based）
    slice_indices = set()
    for image_name in particles_df[image_name_col]:
        # 格式：n@x.mrcs
        if '@' in str(image_name):
            n_str = str(image_name).split('@')[0]
            try:
                n = int(n_str)
                slice_indices.add(n)
            except ValueError:
                # 如果无法解析为整数，跳过
                continue
    
    # 如果指定了输出文件，保存0-based的indices
    if output_index_file is not None:
        output_index_file = Path(output_index_file).resolve()  # 转换为绝对路径
        output_index_file.parent.mkdir(parents=True, exist_ok=True)
        # 转换为0-based并排序
        indices_0based = sorted([n - 1 for n in slice_indices])
        print(f"DEBUG: Writing index file to: {output_index_file}")
        with open(output_index_file, 'w', encoding='utf-8') as f:
            for idx in indices_0based:
                f.write(f"{idx}\n")
        # 验证文件是否真的被写入
        if output_index_file.exists():
            print(f"DEBUG: Index file successfully written: {output_index_file}")
        else:
            print(f"DEBUG: ERROR - Index file not found after writing: {output_index_file}")
    
    return slice_indices


def extract_lines_by_indices(
    txt_file: Union[str, Path],
    indices: Union[List[int], Set[int]],
    output_file: Union[str, Path] = None,
    indices_are_0based: bool = False
) -> List[str]:
    """
    从txt文件中提取指定index对应的行。
    
    Parameters
    ----------
    txt_file : str or Path
        输入的txt文件路径，每行一个条目
    indices : List[int] or Set[int]
        要提取的行index列表
    output_file : str or Path, optional
        输出文件路径。如果为None，则返回行列表但不写入文件
    indices_are_0based : bool, default False
        如果True，indices是0-based的；如果False，indices是1-based的（与star文件中的slice index对应）
        
    Returns
    -------
    List[str]
        提取的行列表（去除换行符）
        
    Examples
    --------
    >>> # 使用1-based indices（从star文件提取的）
    >>> lines = extract_lines_by_indices("subtomos.txt", [1, 3, 5], "output.txt")
    >>> # 使用0-based indices（从txt文件读取的）
    >>> lines = extract_lines_by_indices("subtomos.txt", [0, 2, 4], "output.txt", indices_are_0based=True)
    """
    txt_file = Path(txt_file)
    if not txt_file.exists():
        raise FileNotFoundError(f"Text file not found: {txt_file}")
    
    # 将indices转换为set
    indices_set = set(indices)
    
    # 读取所有行
    with open(txt_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    # 提取指定index的行
    extracted_lines = []
    for idx in sorted(indices_set):
        # 根据indices_are_0based决定是否转换
        if indices_are_0based:
            line_idx = idx  # 已经是0-based
        else:
            line_idx = idx - 1  # 从1-based转换为0-based
        
        if 0 <= line_idx < len(all_lines):
            line = all_lines[line_idx].rstrip('\n\r')  # 去除换行符
            extracted_lines.append(line)
        else:
            # 如果index超出范围，发出警告但继续
            print(f"Warning: Index {idx} (line {line_idx + 1}) is out of range (file has {len(all_lines)} lines)")
    
    # 如果指定了输出文件，写入文件
    if output_file is not None:
        output_file = Path(output_file).resolve()  # 转换为绝对路径
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"DEBUG: Writing output file to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in extracted_lines:
                f.write(line + '\n')
        # 验证文件是否真的被写入
        if output_file.exists():
            print(f"DEBUG: Output file successfully written: {output_file}")
        else:
            print(f"DEBUG: ERROR - Output file not found after writing: {output_file}")
    
    return extracted_lines


def extract_particle_id_from_path(particle_path: str) -> int:
    """
    从颗粒路径的basename中提取颗粒ID。
    
    路径格式：particle_035948.mrc -> 提取出 35948（去掉前导零）
    
    Parameters
    ----------
    particle_path : str
        颗粒文件路径，如 "/path/to/particle_035948.mrc"
        
    Returns
    -------
    int
        颗粒ID，如 35948
        
    Examples
    --------
    >>> extract_particle_id_from_path("/data/particle_035948.mrc")
    35948
    >>> extract_particle_id_from_path("particle_001234.mrc")
    1234
    """
    basename = os.path.basename(particle_path)
    # 移除扩展名
    name_without_ext = os.path.splitext(basename)[0]
    
    # 提取particle_后面的数字部分
    if name_without_ext.startswith('particle_'):
        id_str = name_without_ext[len('particle_'):]
        try:
            # 转换为整数，自动去掉前导零
            particle_id = int(id_str)
            return particle_id
        except ValueError:
            raise ValueError(f"Could not extract particle ID from path: {particle_path}")
    else:
        raise ValueError(f"Path basename does not start with 'particle_': {basename}")


def extract_particle_ids_from_txt(txt_file: Union[str, Path]) -> Set[int]:
    """
    从txt文件中提取所有颗粒ID。
    
    txt文件中每行是一个颗粒路径，格式如：particle_035948.mrc
    
    Parameters
    ----------
    txt_file : str or Path
        包含颗粒路径的txt文件，每行一个路径
        
    Returns
    -------
    Set[int]
        所有唯一的颗粒ID集合
        
    Examples
    --------
    >>> ids = extract_particle_ids_from_txt("particles.txt")
    >>> print(ids)
    {35948, 1234, 5678, ...}
    """
    txt_file = Path(txt_file)
    if not txt_file.exists():
        raise FileNotFoundError(f"Text file not found: {txt_file}")
    
    particle_ids = set()
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                particle_id = extract_particle_id_from_path(line)
                particle_ids.add(particle_id)
            except ValueError as e:
                print(f"Warning: {e}, skipping line: {line}")
                continue
    
    return particle_ids


def _write_dynamo_tbl_from_df(df: pd.DataFrame, output_path: Union[str, Path]):
    """
    将DataFrame写回Dynamo格式的tbl文件。
    
    这是一个内部辅助函数，用于将DataFrame转换回原始tbl格式。
    
    Parameters
    ----------
    df : pd.DataFrame
        包含Dynamo表数据的DataFrame
    output_path : str or Path
        输出tbl文件路径
    """
    output_path = Path(output_path).resolve()  # 转换为绝对路径
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: Writing TBL file to: {output_path}")
    
    # 1-based integer columns per Dynamo convention
    int_cols_1_based = {1, 2, 3, 13, 20, 21, 22, 23, 31, 32, 34, 35}
    
    # 获取列名对应的列索引（1-based）
    col_to_idx = {}
    for i, col_name in enumerate(df.columns, start=1):
        col_to_idx[col_name] = i
    
    with open(output_path, 'w') as fh:
        for _, row in df.iterrows():
            parts = []
            for col_name in df.columns:
                value = row[col_name]
                col_idx = col_to_idx[col_name]
                
                if col_idx in int_cols_1_based:
                    # 整数列
                    if pd.isna(value):
                        parts.append('0')
                    else:
                        parts.append(str(int(round(value))))
                else:
                    # 浮点数列
                    if pd.isna(value):
                        parts.append('0')
                    else:
                        # 使用general format避免尾随零
                        parts.append(format(float(value), '.6g'))
            fh.write(' '.join(parts) + '\n')
    
    # 验证文件是否真的被写入
    if output_path.exists():
        print(f"DEBUG: TBL file successfully written: {output_path}")
    else:
        print(f"DEBUG: ERROR - TBL file not found after writing: {output_path}")


def filter_dynamo_tbl_by_particle_ids(
    tbl_file: Union[str, Path],
    particle_ids: Union[List[int], Set[int]],
    output_tbl_file: Union[str, Path]
) -> pd.DataFrame:
    """
    从Dynamo tbl文件中提取指定颗粒ID对应的行。
    
    tbl文件的第一列（tag列）是颗粒ID，函数会匹配所有与给定颗粒ID相同的行。
    
    Parameters
    ----------
    tbl_file : str or Path
        输入的Dynamo tbl文件路径
    particle_ids : List[int] or Set[int]
        要提取的颗粒ID列表
    output_tbl_file : str or Path
        输出的tbl文件路径
        
    Returns
    -------
    pd.DataFrame
        过滤后的DataFrame，包含匹配的行
        
    Examples
    --------
    >>> filtered_df = filter_dynamo_tbl_by_particle_ids(
    ...     "particles.tbl",
    ...     [35948, 1234, 5678],
    ...     "filtered_particles.tbl"
    ... )
    """
    tbl_file = Path(tbl_file)
    if not tbl_file.exists():
        raise FileNotFoundError(f"TBL file not found: {tbl_file}")
    
    # 读取tbl文件
    df = read_dynamo_tbl(tbl_file)
    
    # 检查是否有tag列（第一列，颗粒ID）
    if 'tag' not in df.columns:
        raise ValueError("TBL file does not contain 'tag' column (first column)")
    
    # 将particle_ids转换为set
    particle_ids_set = set(particle_ids)
    
    # 过滤DataFrame：只保留tag列在particle_ids_set中的行
    filtered_df = df[df['tag'].isin(particle_ids_set)].copy()
    
    if len(filtered_df) == 0:
        print(f"Warning: No matching particles found in TBL file. Requested IDs: {sorted(particle_ids_set)}")
    else:
        print(f"Found {len(filtered_df)} matching particles out of {len(df)} total particles")
        # 检查是否有缺失的ID
        found_ids = set(filtered_df['tag'].unique())
        missing_ids = particle_ids_set - found_ids
        if missing_ids:
            print(f"Warning: {len(missing_ids)} particle IDs not found in TBL file: {sorted(missing_ids)[:10]}...")
    
    # 写入新的tbl文件
    _write_dynamo_tbl_from_df(filtered_df, output_tbl_file)
    
    return filtered_df


def extract_tbl_by_particle_txt(
    txt_file: Union[str, Path],
    tbl_file: Union[str, Path],
    output_tbl_file: Union[str, Path]
) -> pd.DataFrame:
    """
    整合功能：从txt文件中提取颗粒ID，然后从tbl文件中提取对应的行。
    
    这个函数整合了extract_particle_ids_from_txt和filter_dynamo_tbl_by_particle_ids的功能。
    
    Parameters
    ----------
    txt_file : str or Path
        包含颗粒路径的txt文件，每行一个路径（格式：particle_035948.mrc）
    tbl_file : str or Path
        输入的Dynamo tbl文件路径
    output_tbl_file : str or Path
        输出的tbl文件路径
        
    Returns
    -------
    pd.DataFrame
        过滤后的DataFrame，包含匹配的行
        
    Examples
    --------
    >>> filtered_df = extract_tbl_by_particle_txt(
    ...     "particles.txt",
    ...     "all_particles.tbl",
    ...     "filtered_particles.tbl"
    ... )
    """
    # 从txt文件提取颗粒ID
    particle_ids = extract_particle_ids_from_txt(txt_file)
    print(f"Extracted {len(particle_ids)} unique particle IDs from txt file")
    
    # 从tbl文件提取对应的行
    filtered_df = filter_dynamo_tbl_by_particle_ids(
        tbl_file,
        particle_ids,
        output_tbl_file
    )
    
    return filtered_df


def extract_lines_from_star_and_txt(
    star_file: Union[str, Path],
    txt_file: Union[str, Path],
    output_file: Union[str, Path],
    index_file: Union[str, Path] = None
) -> List[str]:
    """
    整合功能：从star文件中提取slice indices，然后从txt文件中提取对应的行。
    
    这个函数整合了extract_slice_indices_from_star和extract_lines_by_indices的功能。
    
    Parameters
    ----------
    star_file : str or Path
        RELION particle star文件路径
    txt_file : str or Path
        输入的txt文件路径，每行一个条目
    output_file : str or Path
        输出txt文件路径，保存提取的行
    index_file : str or Path, optional
        如果提供，将0-based的indices保存到此txt文件（每行一个index，从0开始）
        
    Returns
    -------
    List[str]
        提取的行列表（去除换行符）
        
    Examples
    --------
    >>> lines = extract_lines_from_star_and_txt(
    ...     "particles.star",
    ...     "subtomos.txt",
    ...     "extracted_lines.txt",
    ...     "indices.txt"
    ... )
    """
    # 从star文件提取indices（1-based）
    indices_1based = extract_slice_indices_from_star(star_file, index_file)
    
    # 从txt文件提取对应的行（使用1-based indices）
    extracted_lines = extract_lines_by_indices(
        txt_file,
        indices_1based,
        output_file,
        indices_are_0based=False
    )
    
    return extracted_lines


def process_star_txt_tbl(
    star_file: Union[str, Path],
    particle_txt_file: Union[str, Path],
    tbl_file: Union[str, Path],
    output_dir: Union[str, Path]
) -> dict:
    """
    整合功能：从star文件提取slice indices，从particle txt提取对应行，提取颗粒ID，从tbl提取对应行。
    
    完整流程：
    1. 从star文件提取slice indices（1-based）
    2. 从particle txt文件提取对应的行（根据slice indices）
    3. 从提取的particle txt中提取颗粒ID
    4. 从dynamo tbl文件中提取对应的行
    5. 保存所有结果到输出目录
    
    输出文件命名规则（在输出目录中）：
    - index文件：index.txt
    - particle txt文件：原文件名前加".processed."，如 particles.txt -> particles.processed.txt
    - tbl文件：原文件名前加".processed."，如 all_particles.tbl -> all_particles.processed.tbl
    
    Parameters
    ----------
    star_file : str or Path
        RELION particle star文件路径
    particle_txt_file : str or Path
        输入的颗粒路径txt文件，每行一个路径（格式：particle_035948.mrc）
    tbl_file : str or Path
        输入的Dynamo tbl文件路径
    output_dir : str or Path
        输出目录，所有生成的文件将保存在此目录中
        
    Returns
    -------
    dict
        包含输出文件路径的字典：
        - 'index_file': index.txt路径
        - 'particle_txt_file': 处理后的particle txt文件路径
        - 'tbl_file': 处理后的tbl文件路径
        - 'indices_1based': 提取的1-based indices集合
        - 'particle_ids': 提取的颗粒ID集合
        
    Examples
    --------
    >>> result = process_star_txt_tbl(
    ...     "particles.star",
    ...     "particles.txt",
    ...     "all_particles.tbl",
    ...     "output_dir"
    ... )
    >>> print(result['index_file'])
    output_dir/index.txt
    >>> print(result['particle_txt_file'])
    output_dir/particles.processed.txt
    >>> print(result['tbl_file'])
    output_dir/all_particles.processed.tbl
    """
    star_file = Path(star_file)
    particle_txt_file = Path(particle_txt_file)
    tbl_file = Path(tbl_file)
    output_dir = Path(output_dir)
    
    # 检查输入文件
    if not star_file.exists():
        raise FileNotFoundError(f"Star file not found: {star_file}")
    if not particle_txt_file.exists():
        raise FileNotFoundError(f"Particle txt file not found: {particle_txt_file}")
    if not tbl_file.exists():
        raise FileNotFoundError(f"TBL file not found: {tbl_file}")
    
    # 创建输出目录（转换为绝对路径以确保正确）
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory (absolute): {output_dir}")
    
    # 步骤1：从star文件提取slice indices（1-based）
    index_file = output_dir / "index.txt"
    print(f"Step 1: Saving index file to: {index_file.absolute()}")
    indices_1based = extract_slice_indices_from_star(star_file, index_file)
    print(f"Step 1: Extracted {len(indices_1based)} unique slice indices from star file")
    # 验证文件是否创建
    if index_file.exists():
        print(f"Step 1: ✓ Index file created successfully: {index_file.absolute()}")
    else:
        print(f"Step 1: ✗ WARNING: Index file not found at: {index_file.absolute()}")
    
    # 步骤2：从particle txt文件提取对应的行
    particle_txt_basename = particle_txt_file.stem  # 不含扩展名的文件名
    particle_txt_ext = particle_txt_file.suffix      # 扩展名
    output_particle_txt = output_dir / f"{particle_txt_basename}.processed{particle_txt_ext}"
    print(f"Step 2: Saving particle txt file to: {output_particle_txt.absolute()}")
    
    extracted_lines = extract_lines_by_indices(
        particle_txt_file,
        indices_1based,
        output_particle_txt,
        indices_are_0based=False
    )
    print(f"Step 2: Extracted {len(extracted_lines)} lines from particle txt file")
    # 验证文件是否创建
    if output_particle_txt.exists():
        print(f"Step 2: ✓ Particle txt file created successfully: {output_particle_txt.absolute()}")
    else:
        print(f"Step 2: ✗ WARNING: Particle txt file not found at: {output_particle_txt.absolute()}")
    
    # 步骤3：从提取的particle txt中提取颗粒ID
    particle_ids = extract_particle_ids_from_txt(output_particle_txt)
    print(f"Step 3: Extracted {len(particle_ids)} unique particle IDs")
    
    # 步骤4：从dynamo tbl文件中提取对应的行
    tbl_basename = tbl_file.stem  # 不含扩展名的文件名
    tbl_ext = tbl_file.suffix     # 扩展名
    output_tbl = output_dir / f"{tbl_basename}.processed{tbl_ext}"
    print(f"Step 4: Saving tbl file to: {output_tbl.absolute()}")
    
    filtered_df = filter_dynamo_tbl_by_particle_ids(
        tbl_file,
        particle_ids,
        output_tbl
    )
    print(f"Step 4: Extracted {len(filtered_df)} particles from tbl file")
    # 验证文件是否创建
    if output_tbl.exists():
        print(f"Step 4: ✓ TBL file created successfully: {output_tbl.absolute()}")
    else:
        print(f"Step 4: ✗ WARNING: TBL file not found at: {output_tbl.absolute()}")
    
    return {
        'index_file': index_file,
        'particle_txt_file': output_particle_txt,
        'tbl_file': output_tbl,
        'indices_1based': indices_1based,
        'particle_ids': particle_ids
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='2D投影结果分析工具')
    parser.add_argument('-s', '--star', type=str, help='输入particle star文件路径')
    parser.add_argument('-t', '--txt', type=str, help='输入txt文件路径')
    parser.add_argument('-i', '--indices', type=str, nargs='+', 
                       help='要提取的行index列表（1-based），空格分隔')
    parser.add_argument('-o', '--output', type=str, help='输出txt文件路径')
    parser.add_argument('--index-file', type=str, help='保存0-based indices的txt文件路径')
    parser.add_argument('--zero-based', '--0based', dest='zero_based', action='store_true', 
                       help='indices是0-based的（仅在使用-i参数时有效）')
    parser.add_argument('--tbl', type=str, help='输入Dynamo tbl文件路径')
    parser.add_argument('--output-tbl', type=str, help='输出tbl文件路径')
    parser.add_argument('--extract-tbl-by-txt', action='store_true',
                       help='从txt文件提取颗粒ID，然后从tbl文件提取对应的行')
    parser.add_argument('--output-dir', type=str, help='输出目录（用于process-star-txt-tbl功能）')
    parser.add_argument('--process-star-txt-tbl', action='store_true',
                       help='整合功能：从star提取indices，从particle txt提取行，提取颗粒ID，从tbl提取行')
    
    args = parser.parse_args()
    
    # 如果提供了star文件和txt文件，使用整合函数
    if args.star and args.txt:
        if args.output is None:
            args.output = str(Path(args.txt).parent / f"{Path(args.txt).stem}_extracted.txt")
        lines = extract_lines_from_star_and_txt(
            args.star,
            args.txt,
            args.output,
            args.index_file
        )
        print(f"Extracted {len(lines)} lines from {args.txt} to {args.output}")
        if args.index_file:
            print(f"Saved 0-based indices to {args.index_file}")
    
    # 如果只提供了star文件，只提取indices
    elif args.star:
        indices = extract_slice_indices_from_star(args.star, args.index_file)
        print(f"Found {len(indices)} unique slice indices (1-based):")
        print(sorted(indices))
        if args.index_file:
            print(f"Saved 0-based indices to {args.index_file}")
    
    # 如果只提供了txt和indices，提取指定行
    elif args.txt and args.indices:
        indices = [int(i) for i in args.indices]
        if args.output is None:
            args.output = str(Path(args.txt).parent / f"{Path(args.txt).stem}_extracted.txt")
        lines = extract_lines_by_indices(
            args.txt, 
            indices, 
            args.output,
            indices_are_0based=getattr(args, 'zero_based', False)
        )
        print(f"Extracted {len(lines)} lines to: {args.output}")
    
    # 如果使用process-star-txt-tbl功能（新的整合功能）
    elif args.process_star_txt_tbl:
        if not args.star:
            parser.error("--process-star-txt-tbl requires --star argument")
        if not args.txt:
            parser.error("--process-star-txt-tbl requires --txt argument")
        if not args.tbl:
            parser.error("--process-star-txt-tbl requires --tbl argument")
        if not args.output_dir:
            parser.error("--process-star-txt-tbl requires --output-dir argument")
        
        result = process_star_txt_tbl(
            args.star,
            args.txt,
            args.tbl,
            args.output_dir
        )
        print(f"\n==========================================")
        print(f"Processing completed successfully!")
        print(f"==========================================")
        print(f"Output directory:   {Path(args.output_dir).absolute()}")
        print(f"")
        print(f"Generated files:")
        print(f"  1. Index file:        {result['index_file'].absolute()}")
        print(f"  2. Particle txt file: {result['particle_txt_file'].absolute()}")
        print(f"  3. TBL file:          {result['tbl_file'].absolute()}")
        print(f"")
        print(f"Summary:")
        print(f"  - Extracted {len(result['indices_1based'])} unique slice indices")
        print(f"  - Extracted {len(result['particle_ids'])} unique particle IDs")
        print(f"  - Extracted {len(result['particle_ids'])} particles from TBL")
        print(f"==========================================")
    
    # 如果使用extract-tbl-by-txt功能
    elif args.extract_tbl_by_txt:
        if not args.txt:
            parser.error("--extract-tbl-by-txt requires --txt argument")
        if not args.tbl:
            parser.error("--extract-tbl-by-txt requires --tbl argument")
        if not args.output_tbl:
            args.output_tbl = str(Path(args.tbl).parent / f"{Path(args.tbl).stem}_filtered.tbl")
        
        filtered_df = extract_tbl_by_particle_txt(
            args.txt,
            args.tbl,
            args.output_tbl
        )
        print(f"Extracted {len(filtered_df)} particles to: {args.output_tbl}")
    
    # 如果只提供了txt和tbl，但没有使用extract-tbl-by-txt标志，给出提示
    elif args.txt and args.tbl:
        parser.error("Use --extract-tbl-by-txt flag to extract tbl rows by particle IDs from txt file")
    
    else:
        parser.print_help()

