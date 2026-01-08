import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import starfile

COLUMNS_NAME = {
        1: 'tag', 2: 'aligned', 3: 'averaged',
        4: 'dx', 5: 'dy', 6: 'dz',
        7: 'tdrot', 8: 'tilt', 9: 'narot',
        10: 'cc', 11: 'cc2', 12: 'cpu', 13: 'ftype',
        14: 'ymintilt', 15: 'ymaxtilt', 16: 'xmintilt', 17: 'xmaxtilt',
        18: 'fs1', 19: 'fs2', 20: 'tomo', 21: 'reg', 22: 'class', 23: 'annotation',
        24: 'x', 25: 'y', 26: 'z', 27: 'dshift', 28: 'daxis', 29: 'dnarot', 30: 'dcc',
        31: 'otag', 32: 'npar', 34: 'ref', 35: 'sref'
    }


def convert_euler(
    angles,
    src_convention='ZYZ',
    dst_convention='ZXZ',
    degrees=True
):
    """
    Convert Euler angles between conventions (e.g., ZYZ <-> ZXZ).

    Parameters
    ----------
    angles : array-like
        Shape (N, 3) or (3,) angles.
    src_convention : str
        Source Euler convention, e.g. 'ZYZ', 'ZXZ'.
    dst_convention : str
        Destination Euler convention.
    degrees : bool
        Interpret input and output as degrees when True.

    Returns
    -------
    np.ndarray
        Converted angles with same shape.
    """
    arr = np.asarray(angles, dtype=float)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
        single = True
    rot = R.from_euler(src_convention.upper(), arr, degrees=degrees)
    out = rot.as_euler(dst_convention.upper(), degrees=degrees)
    if single:
        return out.reshape(3,)
    return out


def create_dynamo_table(
    coordinates,
    angles_zyz=None,
    micrograph_names=None,
    output_file='particles.tbl'
):
    """
    Create a Dynamo-format table (.tbl) for subtomogram particles.

    Parameters
    ----------
    coordinates : array-like
        Shape (N, 3) with columns X, Y, Z in voxel units.
    angles_zyz : array-like or None, optional
        Shape (N, 3) with ZYZ Euler angles in degrees: (rotZ, tiltY, psiZ).
        If None, initialized to zeros.
        Convert ZYZ (RELION) -> ZXZ (Dynamo) using reusable converter
    micrograph_names : list[str] or None, optional
        List of tomogram/micrograph names per particle (length N). If None,
        all particles are assigned to tomo id 1. Names are mapped to integer
        tomo ids starting at 1 in first-appearance order.
    output_file : str, optional
        Path to output Dynamo .tbl file. Defaults to 'particles.tbl'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the constructed table columns with names for
        readability. The written .tbl file contains no header.
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('coordinates must have shape (N, 3)')
    num_particles = coords.shape[0]

    if angles_zyz is None:
        angles_zyz_arr = np.zeros((num_particles, 3), dtype=float)
    else:
        angles_zyz_arr = np.asarray(angles_zyz, dtype=float)
        if angles_zyz_arr.shape != (num_particles, 3):
            raise ValueError('angles_zyz must have shape (N, 3) matching coordinates')

    # Convert ZYZ (RELION) -> ZXZ (Dynamo) using reusable converter
    angles_zxz = convert_euler(angles_zyz_arr, src_convention='ZYZ', dst_convention='ZXZ', degrees=True)
    tdrot = angles_zxz[:, 0]
    tilt = angles_zxz[:, 1]
    narot = angles_zxz[:, 2]

    # Map micrograph names to sequential tomo ids starting from 1
    if micrograph_names is None:
        tomo_ids = np.ones((num_particles,), dtype=int)
    else:
        if len(micrograph_names) != num_particles:
            raise ValueError('micrograph_names length must match number of coordinates')
        name_to_id = {}
        next_id = 1
        tomo_ids_list = []
        for name in micrograph_names:
            if name not in name_to_id:
                name_to_id[name] = next_id
                next_id += 1
            tomo_ids_list.append(name_to_id[name])
        tomo_ids = np.asarray(tomo_ids_list, dtype=int)

    # Build Dynamo table columns (1-based index meaning shown in comments)
    # We will output up to column 35 as requested. Missing ones are zeros.
    num_cols = 35
    T = np.zeros((num_particles, num_cols), dtype=float)

    # 1: tag (sequential starting at 1)
    T[:, 0] = np.arange(1, num_particles + 1, dtype=float)
    # 2: aligned (default 1), 3: averaged (default 0)
    T[:, 1] = 1.0
    T[:, 2] = 0.0
    # 4-6: dx, dy, dz (defaults 0)
    # 7-9: tdrot, tilt, narot (ZXZ, degrees)
    T[:, 6] = tdrot
    T[:, 7] = tilt
    T[:, 8] = narot
    # 10-12: cc, cc2, cpu (defaults 0)
    # 13: ftype; 14-17: tilt ranges; 18-19: fs1, fs2 (defaults 0)
    # 20: tomo
    T[:, 19] = tomo_ids.astype(float)
    # 21: reg, 22: class, 23: annotation (defaults 0)
    # 24-26: x, y, z coordinates
    T[:, 23] = coords[:, 0]
    T[:, 24] = coords[:, 1]
    T[:, 25] = coords[:, 2]
    # 27-32: dshift, daxis, dnarot, dcc, otag, npar (defaults 0)
    # 34: ref (default 1), 35: sref (default 0)
    T[:, 33] = 1.0

    # Prepare a labeled DataFrame for return (note: file has no header)

    col_names = []
    for i in range(1, num_cols + 1):
        col_names.append(COLUMNS_NAME.get(i, f'col{i}'))
    df = pd.DataFrame(T, columns=col_names)

    # Write ASCII .tbl without header using Dynamo-like formatting
    # Integer-like columns are written as integers; others as compact floats
    def _write_dynamo_tbl(table_array, path):
        # 1-based integer columns per Dynamo convention
        int_cols_1_based = {1, 2, 3, 13, 20, 21, 22, 23, 31, 32, 34, 35}
        int_cols_0_based = {idx - 1 for idx in int_cols_1_based}

        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(path, 'w') as fh:
            for row in table_array:
                parts = []
                for j, value in enumerate(row):
                    if j in int_cols_0_based:
                        parts.append(str(int(round(value))))
                    else:
                        # Use general format to avoid trailing zeros while
                        # keeping reasonable precision similar to template
                        parts.append(format(float(value), '.6g'))
                fh.write(' '.join(parts) + '\n')

    _write_dynamo_tbl(T, output_file)

    return df



def read_vll_to_df(vll_path):
    """
    Read a VLL file where each line is a path to an MRC file and
    return a DataFrame with two columns:
      - 'rlnMicrographName': the basename without extension
      - 'tomo_path': the original (stripped) line

    Parameters
    ----------
    vll_path : str
        Path to the .vll file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns 'rlnMicrographName' and 'tomo_path'.
    """
    names = []
    tomo_paths = []
    with open(vll_path, 'r') as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith('#'):
                continue
            base = os.path.basename(p)
            noext, _ = os.path.splitext(base)
            names.append(noext)
            tomo_paths.append(p)
    return pd.DataFrame({'rlnMicrographName': names, 'tomo_path': tomo_paths})


def read_dynamo_tbl(tbl_path, vll_path=None):
    """
    Read a Dynamo .tbl file and optionally a .vll file. Returns a DataFrame
    with Dynamo-defined keys. If vll_path is provided, column 20 (tomo) values
    are mapped to the micrograph basenames (no extension) and the column is
    renamed to 'rlnMicrographName'. If vll_path is None, column 20 remains
    the numeric tomo index.

    Parameters
    ----------
    tbl_path : str
        Path to Dynamo .tbl file (ASCII)
    vll_path : str or None
        Optional path to .vll mapping file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with named columns up to those present in the file.
    """
    # Dynamo tables may contain complex-like tokens such as '0+1.2074e-06i'.
    # We parse line-by-line and convert any such tokens to their real part.
    rows = []
    max_cols = 0
    with open(tbl_path, 'r') as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            tokens = line.split()
            parsed = []
            for tok in tokens:
                s = tok.replace('I', 'i')
                if 'i' in s or 'j' in s:
                    # Convert Dynamo-style imaginary to Python complex and take real part
                    s_complex = s.replace('i', 'j')
                    try:
                        parsed.append(float(complex(s_complex).real))
                    except Exception:
                        # Fallback: strip at '+' and keep the left real part
                        left = s.split('+', 1)[0]
                        try:
                            parsed.append(float(left))
                        except Exception:
                            parsed.append(np.nan)
                else:
                    try:
                        parsed.append(float(s))
                    except Exception:
                        parsed.append(np.nan)
            rows.append(parsed)
            if len(parsed) > max_cols:
                max_cols = len(parsed)

    # Normalize row lengths by padding with NaN if necessary
    normalized = []
    for r in rows:
        if len(r) < max_cols:
            r = r + [np.nan] * (max_cols - len(r))
        normalized.append(r)
    data = np.asarray(normalized, dtype=float)
    ncols = data.shape[1]
    # Build column names using COLUMNS_NAME, fallback to col{i}
    col_names = [COLUMNS_NAME.get(i, f'col{i}') for i in range(1, ncols + 1)]
    df = pd.DataFrame(data, columns=col_names)

    if vll_path is not None and 'tomo' in df.columns:
        vll_df = read_vll_to_df(vll_path)
        # Map 1..len(vll) to names in order
        mapping = {i + 1: vll_df['rlnMicrographName'].iloc[i]
                   for i in range(len(vll_df))}
        # Replace values; if out of range, keep original as string
        mapped = df['tomo'].round().astype(int).map(mapping)
        # Fallback for ids not in mapping
        mapped = mapped.fillna(df['tomo'].astype(int).astype(str))
        df = df.drop(columns=['tomo'])
        df.insert(19, 'rlnMicrographName', mapped.values)
    return df


def dynamo_df_to_relion(df, bin_scalar=8.0, pixel_size=None, tomogram_size=None, output_centered=True):
    """
    Transform a DataFrame from read_dynamo_tbl into a RELION-like DataFrame.
    
    IMPORTANT COORDINATE SYSTEM NOTES:
    - Dynamo's x/y/z (columns 24-26) are ABSOLUTE coordinates from the origin,
      in pixel units (typically binned).
    - RELION's rlnCenteredCoordinateXAngst/Y/Z are coordinates RELATIVE TO THE
      TOMOGRAM CENTER, in Angstrom units.
    
    If output_centered=True (default), conversion process:
    1. Get absolute coordinates from Dynamo table (multiply by bin_scalar if needed)
    2. Convert to centered coordinates: centered = absolute - (tomogram_size / 2)
    3. Convert to Angstrom: centered_angstrom = centered_pixels * pixel_size
    
    If output_centered=False, outputs rlnCoordinateX/Y/Z (absolute coordinates in pixels).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame produced by read_dynamo_tbl.
    bin_scalar : float
        Scalar to multiply coordinates. Default is 8.0 for 8x8x8 binning.
        This converts binned coordinates to unbinned pixel coordinates.
    pixel_size : float or None, optional
        Pixel size in Angstrom. Required if output_centered=True.
        Used to convert pixel coordinates to Angstrom.
    tomogram_size : tuple or array-like of shape (3,) or None, optional
        Tomogram dimensions as (size_x, size_y, size_z) in UNBINNED pixels.
        Required if output_centered=True. This should be the size AFTER applying bin_scalar.
    output_centered : bool, optional
        If True (default), output rlnCenteredCoordinateXAngst/Y/Z (relative to center, in Angstrom).
        If False, output rlnCoordinateX/Y/Z (absolute, in pixels).

    Returns
    -------
    pandas.DataFrame
        RELION-style DataFrame with coordinates and angles.
    """
    # Get absolute coordinates from Dynamo table (in binned pixels)
    x_binned = df['x'] if 'x' in df.columns else df[COLUMNS_NAME.get(24, 'x')]
    y_binned = df['y'] if 'y' in df.columns else df[COLUMNS_NAME.get(25, 'y')]
    z_binned = df['z'] if 'z' in df.columns else df[COLUMNS_NAME.get(26, 'z')]
    
    # Convert to unbinned pixel coordinates
    x_unbinned = x_binned * float(bin_scalar)
    y_unbinned = y_binned * float(bin_scalar)
    z_unbinned = z_binned * float(bin_scalar)
    
    # Convert to centered coordinates if requested
    if output_centered:
        if pixel_size is None:
            raise ValueError("pixel_size is required when output_centered=True")
        if tomogram_size is None:
            raise ValueError("tomogram_size is required when output_centered=True")
        
        tomogram_size = np.asarray(tomogram_size, dtype=float)
        if tomogram_size.shape != (3,):
            raise ValueError(f"tomogram_size must be shape (3,), got {tomogram_size.shape}")
        
        # Calculate tomogram center
        tomogram_center = tomogram_size / 2.0
        
        # Convert absolute to centered coordinates
        x_centered_pixels = x_unbinned.values - tomogram_center[0]
        y_centered_pixels = y_unbinned.values - tomogram_center[1]
        z_centered_pixels = z_unbinned.values - tomogram_center[2]
        
        # Convert to Angstrom
        x_angstrom = x_centered_pixels * float(pixel_size)
        y_angstrom = y_centered_pixels * float(pixel_size)
        z_angstrom = z_centered_pixels * float(pixel_size)
        
        coord_x_col = 'rlnCenteredCoordinateXAngst'
        coord_y_col = 'rlnCenteredCoordinateYAngst'
        coord_z_col = 'rlnCenteredCoordinateZAngst'
        x_out = x_angstrom
        y_out = y_angstrom
        z_out = z_angstrom
    else:
        coord_x_col = 'rlnCoordinateX'
        coord_y_col = 'rlnCoordinateY'
        coord_z_col = 'rlnCoordinateZ'
        x_out = x_unbinned.values
        y_out = y_unbinned.values
        z_out = z_unbinned.values

    # Angles ZXZ -> ZYZ
    tdrot = df['tdrot'] if 'tdrot' in df.columns else df[COLUMNS_NAME.get(7, 'tdrot')]
    tilt = df['tilt'] if 'tilt' in df.columns else df[COLUMNS_NAME.get(8, 'tilt')]
    narot = df['narot'] if 'narot' in df.columns else df[COLUMNS_NAME.get(9, 'narot')]
    angles_zxz = np.stack([tdrot.values, tilt.values, narot.values], axis=1)
    angles_zyz = convert_euler(angles_zxz, src_convention='ZXZ', dst_convention='ZYZ', degrees=True)

    # Micrograph name
    if 'rlnMicrographName' in df.columns:
        print("Using rlnMicrographName column")
        names = df['rlnMicrographName'].astype(str)
    elif 'tomo' in df.columns:
        print("Using tomo column")
        names = df['tomo'].astype(int).astype(str)
    else:
        # As a fallback, fill with '1'
        names = pd.Series(['1'] * len(df))

    out = pd.DataFrame({
        coord_x_col: x_out,
        coord_y_col: y_out,
        coord_z_col: z_out,
        'rlnAngleRot': angles_zyz[:, 0],
        'rlnAngleTilt': angles_zyz[:, 1],
        'rlnAnglePsi': angles_zyz[:, 2],
        'rlnOriginXAngst': 0.0,
        'rlnOriginYAngst': 0.0,
        'rlnOriginZAngst': 0.0,
        'rlnMicrographName': names.values
    })
    return out


def relion_star_to_dynamo_tbl(star_path, pixel_size, tomogram_size=None, output_file='particles.tbl'):
    """
    Read a RELION particle STAR file and convert it to a Dynamo .tbl file.

    IMPORTANT COORDINATE SYSTEM NOTES:
    - RELION's rlnCenteredCoordinateXAngst/Y/Z are coordinates RELATIVE TO THE
      TOMOGRAM CENTER, in Angstrom units.
    - Dynamo's x/y/z (columns 24-26) are ABSOLUTE coordinates from the origin
      (typically top-left corner), in pixel units.
    
    Conversion process:
    1. Convert CenteredCoordinate from Angstrom to pixels (divide by pixel_size)
    2. Add tomogram center offset to get absolute coordinates:
       absolute_coord = centered_coord_pixels + (tomogram_size / 2)
    
    If tomogram_size is not provided, the function assumes coordinates are already
    absolute (which may be incorrect if using CenteredCoordinate fields).

    Parameters
    ----------
    star_path : str
        Path to RELION particle STAR file.
    pixel_size : float
        Pixel size in Angstrom. Coordinates in Angstrom will be divided by
        this value to convert to pixels.
    tomogram_size : tuple or array-like of shape (3,) or None, optional
        Tomogram dimensions as (size_x, size_y, size_z) in pixels.
        If None, conversion assumes coordinates are already absolute (not recommended).
        If provided, coordinates will be converted from centered to absolute.
    output_file : str, optional
        Path to output Dynamo .tbl file. Defaults to 'particles.tbl'.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the constructed Dynamo table with named columns.
        The .tbl file is also written to disk.

    Examples
    --------
    >>> # With tomogram size (recommended)
    >>> df = relion_star_to_dynamo_tbl('particles.star', pixel_size=1.32, 
    ...                                tomogram_size=(512, 512, 200), 
    ...                                output_file='output.tbl')
    >>> # Without tomogram size (assumes absolute coordinates - may be incorrect)
    >>> df = relion_star_to_dynamo_tbl('particles.star', pixel_size=1.32, 
    ...                                output_file='output.tbl')
    """
    # Read RELION star file
    # starfile.read may return a DataFrame or a dict with multiple blocks
    star_data = starfile.read(star_path, always_dict=False)
    
    # Handle both DataFrame and dict cases
    particles_df = None
    optics_df = None
    
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
                raise ValueError(f"Could not find DataFrame in STAR file dict: {star_path}")
        else:
            raise ValueError(f"Could not find particle data block in STAR file: {star_path}")
        
        # Try to get optics block for tomogram size
        if 'optics' in star_data:
            optics_df = star_data['optics']
    elif isinstance(star_data, pd.DataFrame):
        particles_df = star_data
    else:
        raise ValueError(f"Unexpected data type from starfile.read: {type(star_data)}")

    df = particles_df

    # Required column names
    coord_x_col = 'rlnCenteredCoordinateXAngst'
    coord_y_col = 'rlnCenteredCoordinateYAngst'
    coord_z_col = 'rlnCenteredCoordinateZAngst'
    angle_rot_col = 'rlnAngleRot'
    angle_tilt_col = 'rlnAngleTilt'
    angle_psi_col = 'rlnAnglePsi'
    
    # Check if required columns exist
    missing_cols = []
    for col in [coord_x_col, coord_y_col, coord_z_col, angle_rot_col, angle_tilt_col, angle_psi_col]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(f"Missing required columns in STAR file: {missing_cols}")
    
    # Extract coordinates (in Angstrom, relative to tomogram center)
    coords_angstrom_centered = np.stack([
        df[coord_x_col].values,
        df[coord_y_col].values,
        df[coord_z_col].values
    ], axis=1)
    
    # Convert from Angstrom to pixels (still relative to center)
    coords_pixels_centered = coords_angstrom_centered / float(pixel_size)
    
    # Convert from centered coordinates to absolute coordinates
    if tomogram_size is not None:
        tomogram_size = np.asarray(tomogram_size, dtype=float)
        if tomogram_size.shape != (3,):
            raise ValueError(f"tomogram_size must be shape (3,), got {tomogram_size.shape}")
        # Calculate tomogram center
        tomogram_center = tomogram_size / 2.0
        # Convert to absolute coordinates: absolute = centered + center
        coords_pixels = coords_pixels_centered + tomogram_center
    else:
        # Try to get tomogram size from optics block if available
        if optics_df is not None:
            if 'rlnImageSizeX' in optics_df.columns and 'rlnImageSizeY' in optics_df.columns and 'rlnImageSizeZ' in optics_df.columns:
                tomogram_size = np.array([
                    optics_df['rlnImageSizeX'].iloc[0],
                    optics_df['rlnImageSizeY'].iloc[0],
                    optics_df['rlnImageSizeZ'].iloc[0]
                ])
                tomogram_center = tomogram_size / 2.0
                coords_pixels = coords_pixels_centered + tomogram_center
            else:
                import warnings
                warnings.warn(
                    "WARNING: tomogram_size not provided and could not be extracted from optics block. "
                    "Assuming coordinates are already absolute. This may be INCORRECT if using "
                    "rlnCenteredCoordinate fields. Please provide tomogram_size parameter.",
                    UserWarning
                )
                coords_pixels = coords_pixels_centered
        else:
            import warnings
            warnings.warn(
                "WARNING: tomogram_size not provided. Assuming coordinates are already absolute. "
                "This may be INCORRECT if using rlnCenteredCoordinate fields. "
                "Please provide tomogram_size parameter.",
                UserWarning
            )
            coords_pixels = coords_pixels_centered
    
    # Extract angles (already in ZYZ convention for RELION)
    angles_zyz = np.stack([
        df[angle_rot_col].values,
        df[angle_tilt_col].values,
        df[angle_psi_col].values
    ], axis=1)
    
    # Extract micrograph names if available
    micrograph_names = None
    if 'rlnMicrographName' in df.columns:
        micrograph_names = df['rlnMicrographName'].astype(str).tolist()
    elif 'rlnTomoName' in df.columns:
        micrograph_names = df['rlnTomoName'].astype(str).tolist()
    else:
        raise ValueError(f"Could not find micrograph name column in STAR file: {star_path}")
    
    # Create Dynamo table
    dynamo_df = create_dynamo_table(
        coordinates=coords_pixels,
        angles_zyz=angles_zyz,
        micrograph_names=micrograph_names,
        output_file=output_file
    )
    
    return dynamo_df


def save_sorted_vll_by_tomonames(tomo_names:list, vll_df:pd.DataFrame, output_vll_path:str='sorted_tomos_bin8.vll'):
    """
    Reorder and save a .vll file according to the order of tomo_names using the 'tomo_path' column in vll_df.
    Matching rule: A match is found if the basename (without extension) of an mrc path contains the tomo_name string.

    Parameters
    ----------
    tomo_names : list
        List of tomogram names.
    vll_df : pd.DataFrame
        DataFrame with 'tomo_path' column. Output from read_vll_to_df.
    output_vll_path : str, optional
        Path to output .vll file. Defaults to 'sorted_tomos_bin8.vll'.

    """
    import os
    mrc_paths = vll_df['tomo_path'].tolist()
    # Build a dictionary mapping from each basename (mrc filename without extension) to its path
    basename_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in mrc_paths}

    # For each tomo_name, find the first mrc path whose basename contains tomo_name
    sorted_paths = []
    for name in tomo_names:
        found_path = None
        for bn, p in basename_to_path.items():
            if name in bn:
                found_path = p
                break
        if found_path is None:
            raise ValueError(f"Could not find an mrc path matching {name} (by inclusion rule)")
        sorted_paths.append(found_path)

    # Write the new vll file
    os.makedirs(os.path.dirname(output_vll_path), exist_ok=True)
    with open(output_vll_path, 'w') as f:
        for p in sorted_paths:
            f.write(p + '\n')
    print(f"Saved vll file sorted by tomo_names to: {output_vll_path}")


def dynamo_tbl_vll_to_relion_star(
    tbl_path,
    vll_path=None,
    output_file='particles.star',
    bin_scalar=8.0,
    pixel_size=None,
    tomogram_size=None,
    output_centered=True
):
    """
    将 Dynamo 的 .tbl 文件和可选的 .vll 文件转换为 RELION 格式的 .star 文件。
    
    该函数整合了读取、转换和写入的完整流程：
    1. 读取 Dynamo .tbl 文件（如果提供了 .vll 文件，会映射 tomogram 名称）
    2. 将 Dynamo 格式转换为 RELION 格式（包括坐标系统和欧拉角转换）
    3. 写入 RELION .star 文件
    
    IMPORTANT COORDINATE SYSTEM NOTES:
    - Dynamo's x/y/z (columns 24-26) are ABSOLUTE coordinates from the origin,
      in pixel units (typically binned).
    - RELION's rlnCenteredCoordinateXAngst/Y/Z are coordinates RELATIVE TO THE
      TOMOGRAM CENTER, in Angstrom units.
    
    If output_centered=True (default), conversion process:
    1. Get absolute coordinates from Dynamo table (multiply by bin_scalar if needed)
    2. Convert to centered coordinates: centered = absolute - (tomogram_size / 2)
    3. Convert to Angstrom: centered_angstrom = centered_pixels * pixel_size
    
    If output_centered=False, outputs rlnCoordinateX/Y/Z (absolute coordinates in pixels).
    
    Parameters
    ----------
    tbl_path : str
        Path to Dynamo .tbl file (ASCII format).
    vll_path : str or None, optional
        Optional path to .vll file containing tomogram paths.
        If provided, tomogram IDs in the .tbl file will be mapped to micrograph names.
    output_file : str, optional
        Path to output RELION .star file. Defaults to 'particles.star'.
    bin_scalar : float, optional
        Scalar to multiply coordinates. Default is 8.0 for 8x8x8 binning.
        This converts binned coordinates to unbinned pixel coordinates.
    pixel_size : float or None, optional
        Pixel size in Angstrom. Required if output_centered=True.
        Used to convert pixel coordinates to Angstrom.
    tomogram_size : tuple or array-like of shape (3,) or None, optional
        Tomogram dimensions as (size_x, size_y, size_z) in UNBINNED pixels.
        Required if output_centered=True. This should be the size AFTER applying bin_scalar.
    output_centered : bool, optional
        If True (default), output rlnCenteredCoordinateXAngst/Y/Z (relative to center, in Angstrom).
        If False, output rlnCoordinateX/Y/Z (absolute, in pixels).
    
    Returns
    -------
    pandas.DataFrame
        RELION-style DataFrame containing the converted particle data.
        The .star file is also written to disk.
    
    Examples
    --------
    >>> # Basic conversion with centered coordinates (requires pixel_size and tomogram_size)
    >>> df = dynamo_tbl_vll_to_relion_star(
    ...     'particles.tbl',
    ...     vll_path='tomograms.vll',
    ...     output_file='particles.star',
    ...     bin_scalar=8.0,
    ...     pixel_size=1.32,
    ...     tomogram_size=(512, 512, 200)
    ... )
    >>> 
    >>> # Conversion without vll file (uses numeric tomogram IDs)
    >>> df = dynamo_tbl_vll_to_relion_star(
    ...     'particles.tbl',
    ...     output_file='particles.star',
    ...     bin_scalar=8.0,
    ...     pixel_size=1.32,
    ...     tomogram_size=(512, 512, 200)
    ... )
    >>> 
    >>> # Conversion with absolute coordinates (no pixel_size or tomogram_size needed)
    >>> df = dynamo_tbl_vll_to_relion_star(
    ...     'particles.tbl',
    ...     vll_path='tomograms.vll',
    ...     output_file='particles.star',
    ...     bin_scalar=8.0,
    ...     output_centered=False
    ... )
    """
    # Read Dynamo table (with optional vll mapping)
    dynamo_df = read_dynamo_tbl(tbl_path, vll_path=vll_path)
    
    # Convert to RELION format
    relion_df = dynamo_df_to_relion(
        dynamo_df,
        bin_scalar=bin_scalar,
        pixel_size=pixel_size,
        tomogram_size=tomogram_size,
        output_centered=output_centered
    )
    
    # Write to STAR file
    directory = os.path.dirname(output_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    starfile.write(relion_df, output_file)
    
    print(f"Converted {len(relion_df)} particles from Dynamo format to RELION format.")
    print(f"Output written to: {output_file}")
    
    return relion_df
