import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

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
    return a DataFrame with a single column 'rlnMicrographName',
    containing the basename without extension.

    Parameters
    ----------
    vll_path : str
        Path to the .vll file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with column 'rlnMicrographName'.
    """
    names = []
    with open(vll_path, 'r') as f:
        for line in f:
            p = line.strip()
            if not p or p.startswith('#'):
                continue
            base = os.path.basename(p)
            noext, _ = os.path.splitext(base)
            names.append(noext)
    return pd.DataFrame({'rlnMicrographName': names})


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


def dynamo_df_to_relion(df, bin_scalar=8.0):
    """
    Transform a DataFrame from read_dynamo_tbl into a RELION-like DataFrame
    with keys: 'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
    'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 'rlnMicrographName'.

    Coordinates are multiplied by bin_scalar. Angles are converted from
    Dynamo ZXZ (tdrot, tilt, narot) to RELION ZYZ (rot, tilt, psi).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame produced by read_dynamo_tbl.
    bin_scalar : float
        Scalar to multiply coordinates. Default is 8.0 for 8x8x8 binning.

    Returns
    -------
    pandas.DataFrame
        RELION-style DataFrame suitable for passing to io_starfile.create_relion_particles.
    """
    # Coordinates
    x = (df['x'] if 'x' in df.columns else df[COLUMNS_NAME.get(24, 'x')]) * float(bin_scalar)
    y = (df['y'] if 'y' in df.columns else df[COLUMNS_NAME.get(25, 'y')]) * float(bin_scalar)
    z = (df['z'] if 'z' in df.columns else df[COLUMNS_NAME.get(26, 'z')]) * float(bin_scalar)

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
        'rlnCoordinateX': x.values,
        'rlnCoordinateY': y.values,
        'rlnCoordinateZ': z.values,
        'rlnAngleRot': angles_zyz[:, 0],
        'rlnAngleTilt': angles_zyz[:, 1],
        'rlnAnglePsi': angles_zyz[:, 2],
        'rlnOriginXAngst':0.0,
        'rlnOriginYAngst':0.0,
        'rlnOriginZAngst':0.0,
        'rlnMicrographName': names.values
    })
    return out

