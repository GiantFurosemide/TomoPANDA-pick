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

命令行使用:
    # 从star文件提取indices并保存（0-based）
    python utils/2d_projection/analyze_results.py -s particles.star --index-file indices.txt
    
    # 整合功能：从star和txt提取行
    python utils/2d_projection/analyze_results.py -s particles.star -t subtomos.txt -o output.txt --index-file indices.txt
    
    # 从txt文件提取指定行（使用1-based indices）
    python utils/2d_projection/analyze_results.py -t subtomos.txt -i 1 3 5 -o output.txt
    
    # 从txt文件提取指定行（使用0-based indices）
    python utils/2d_projection/analyze_results.py -t subtomos.txt -i 0 2 4 -o output.txt --zero-based

注意事项：
- star文件中的slice index（n@x.mrcs中的n）是从1开始的（1-based）
- 保存到txt文件的indices会自动转换为0-based（n-1）
- 从txt文件读取行时，默认使用1-based indices（与star文件对应）
- 如果使用--zero-based参数，则使用0-based indices
"""

import starfile
from pathlib import Path
from typing import Set, List, Union


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
    
    # 读取star文件
    star_data = starfile.read(star_file)
    
    # 如果star文件包含多个block（如optics和particles），取particles block
    if isinstance(star_data, dict):
        if 'particles' in star_data:
            particles_df = star_data['particles']
        elif 'data_particles' in star_data:
            particles_df = star_data['data_particles']
        else:
            # 尝试找到包含_rlnImageName的DataFrame
            particles_df = None
            for key, df in star_data.items():
                if isinstance(df, starfile.DataFrame) and '_rlnImageName' in df.columns:
                    particles_df = df
                    break
            if particles_df is None:
                raise ValueError("Could not find particles block in star file")
    else:
        particles_df = star_data
    
    # 检查是否有_rlnImageName列
    if '_rlnImageName' not in particles_df.columns:
        raise ValueError("Star file does not contain '_rlnImageName' column")
    
    # 提取所有的slice index（1-based）
    slice_indices = set()
    for image_name in particles_df['_rlnImageName']:
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
        output_index_file = Path(output_index_file)
        output_index_file.parent.mkdir(parents=True, exist_ok=True)
        # 转换为0-based并排序
        indices_0based = sorted([n - 1 for n in slice_indices])
        with open(output_index_file, 'w', encoding='utf-8') as f:
            for idx in indices_0based:
                f.write(f"{idx}\n")
    
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
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in extracted_lines:
                f.write(line + '\n')
    
    return extracted_lines


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
    
    else:
        parser.print_help()

