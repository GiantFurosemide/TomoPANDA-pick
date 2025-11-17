import pandas as pd
import starfile
import numpy as np



RELION_LABELS = {
    'coordinates': ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'],
    'angles': ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'],
    'micrograph': ['rlnMicrographName'],
    'image': ['rlnImageName'],
    'class': ['rlnClassNumber'],
    'defocus': ['rlnDefocusU', 'rlnDefocusV', 'rlnDefocusAngle'],
    'ctf': ['rlnCtfFigureOfMerit', 'rlnCtfMaxResolution'],
    'particle': ['rlnParticleName', 'rlnGroupNumber']
}

def create_relion_particles(coordinates, angles=None, micrograph_name=None, output_file='particles.star'):
    """
    Create a RELION-format particles STAR file.

    Parameters
    ----------
    coordinates : array-like
        Coordinates array with shape (N, 3), columns correspond to X, Y, Z.
    angles : array-like or None, optional
        Rotation angles array with shape (N, 3); columns correspond to Rot, Tilt, Psi. If None, initialized to zeros.
    micrograph_name : str or list or None, optional
        The micrograph name(s) to assign to all particles, or list of the same length as coordinates.
        If None, defaults to 'micrograph.mrc'.
    output_file : str, optional
        Path to output STAR file. Defaults to 'particles.star'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing generated particles table.

    Examples
    --------
    >>> coords = [[1, 2, 3], [4, 5, 6]]
    >>> angles = [[10, 20, 30], [40, 50, 60]]
    >>> create_relion_particles(coords, angles, "mic1.mrc", "particles.star")
    """
    df = pd.DataFrame(coordinates, columns=RELION_LABELS['coordinates'])
    if angles is not None:
        df[RELION_LABELS['angles']] = angles
    else:
        df[RELION_LABELS['angles']] = 0.0

    if micrograph_name is not None:
        df[RELION_LABELS['micrograph']] = micrograph_name  # list or string
    else:
        df[RELION_LABELS['micrograph']] = 'micrograph.mrc'

    starfile.write(df, output_file)

    return df

def create_relion_optics(image_size, pixel_size, output_file='optics.star'):
    """
    Create a RELION-format optics STAR file.

    Parameters
    ----------
    image_size : int or list or array-like
        Image size for X, Y, Z. Can be scalar (applies to all) or sequence.
    pixel_size : float or list or array-like
        Pixel size (nm or Å).
    output_file : str, optional
        Path to output STAR file. Defaults to 'optics.star'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing optics information.

    Examples
    --------
    >>> create_relion_optics(512, 1.32, "optics.star")
    """
    df = pd.DataFrame({
        'rlnImageSizeX': image_size,
        'rlnImageSizeY': image_size,
        'rlnImageSizeZ': image_size,
        'rlnPixelSize': pixel_size
    })
    starfile.write(df, output_file)
    return df

def create_relion_micrographs(micrograph_names, output_file='micrographs.star'):
    """
    Create a RELION-format micrographs STAR file.

    Parameters
    ----------
    micrograph_names : list or array-like
        List of micrograph file names.
    output_file : str, optional
        Path to output STAR file. Defaults to 'micrographs.star'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 'rlnMicrographName' column.

    Examples
    --------
    >>> create_relion_micrographs(["a.mrc", "b.mrc"], "micrographs.star")
    """
    df = pd.DataFrame({
        'rlnMicrographName': micrograph_names
    })

    starfile.write(df, output_file)
    return df


def create_relion3_star(
    image_names: list,
    angles: np.ndarray,
    optics_group_id: int,
    optics_group_name: str,
    pixel_size: float,
    voltage_kv: float,
    cs_mm: float,
    amplitude_contrast: float,
    output_file: str = 'particles.star'
) -> dict:
    """
    创建 RELION 3 格式的 STAR 文件，包含 optics 和 particles 两个 block。
    
    Parameters
    ----------
    image_names : list
        图像名称列表，格式为 "slice_index@stack_path"
    angles : np.ndarray
        欧拉角数组，形状为 (N, 3)，列为 (Rot, Tilt, Psi)（度）
    optics_group_id : int
        Optics group ID
    optics_group_name : str
        Optics group 名称
    pixel_size : float
        像素大小（Angstrom）
    voltage_kv : float
        电压（kV）
    cs_mm : float
        球差（mm）
    amplitude_contrast : float
        振幅对比度
    output_file : str, optional
        输出文件路径，默认 'particles.star'
        
    Returns
    -------
    dict
        包含 'optics' 和 'particles' DataFrame 的字典
    """
    # 创建 optics DataFrame
    optics_df = pd.DataFrame({
        'rlnOpticsGroupName': [optics_group_name],
        'rlnOpticsGroup': [optics_group_id],
        'rlnImagePixelSize': [pixel_size],
        'rlnVoltage': [voltage_kv],
        'rlnSphericalAberration': [cs_mm],
        'rlnAmplitudeContrast': [amplitude_contrast]
    })
    
    # 创建 particles DataFrame
    particles_df = pd.DataFrame({
        'rlnImageName': image_names,
        'rlnAngleRot': angles[:, 0],
        'rlnAngleTilt': angles[:, 1],
        'rlnAnglePsi': angles[:, 2],
        'rlnOpticsGroup': [optics_group_id] * len(image_names)
    })
    
    # 写入 STAR 文件（包含两个 block）
    starfile.write({'optics': optics_df, 'particles': particles_df}, output_file)
    
    return {'optics': optics_df, 'particles': particles_df}


