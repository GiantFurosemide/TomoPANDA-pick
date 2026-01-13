import numpy as np
from eulerangles import convert_eulers


def convert_euler(
    angles,
    src_convention='ZYZ',
    dst_convention='ZXZ',
    degrees=True,
    intrinsic=True
):
    """
    Convert Euler angles between different conventions (e.g., ZYZ <-> ZXZ).

    This function uses the eulerangles package to convert between Euler angle
    conventions. For RELION <-> Dynamo conversions, it uses the built-in
    'dynamo' and 'relion' metadata. For other conventions, it attempts to
    use the convention names directly.

    Parameters
    ----------
    angles : array-like
        Shape (N, 3) or (3,) Euler angles.
    src_convention : str
        Source Euler convention. Can be:
        - 'ZYZ', 'ZXZ', 'XYZ', etc. (3-character string specifying rotation axes)
        - 'relion' (for RELION ZYZ convention)
        - 'dynamo' (for Dynamo ZXZ convention)
    dst_convention : str
        Destination Euler convention. Same format as src_convention.
    degrees : bool
        Interpret input and output as degrees when True, radians when False.
        Note: eulerangles convert_eulers always works in degrees.
    intrinsic : bool
        If True, use intrinsic rotations (rotations about body-fixed axes).
        If False, use extrinsic rotations (rotations about space-fixed axes).
        Default is True (intrinsic).
        Note: This parameter is kept for API compatibility but may not be
        used when source_meta/target_meta are specified.

    Returns
    -------
    np.ndarray
        Converted angles with same shape as input.

    Examples
    --------
    >>> # Convert RELION ZYZ to Dynamo ZXZ
    >>> relion_angles = np.array([-77.952, 70.291, 46.556])
    >>> dynamo_angles = convert_euler(relion_angles, 'relion', 'dynamo', degrees=True)
    
    >>> # Convert Dynamo ZXZ to RELION ZYZ
    >>> dynamo_angles = np.array([-43.444, 70.291, 12.048])
    >>> relion_angles = convert_euler(dynamo_angles, 'dynamo', 'relion', degrees=True)
    
    >>> # Convert between generic conventions (if supported)
    >>> angles_zyz = np.array([10, 20, 30])
    >>> angles_zxz = convert_euler(angles_zyz, 'ZYZ', 'ZXZ', degrees=True)
    """
    arr = np.asarray(angles, dtype=float)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
        single = True
    
    # Map convention names to eulerangles metadata format
    # Normalize to lowercase for comparison
    src_lower = src_convention.lower()
    dst_lower = dst_convention.lower()
    
    # Map ZYZ/ZXZ to relion/dynamo for convenience
    convention_map = {
        'zyz': 'relion',
        'zxz': 'dynamo'
    }
    
    # Get metadata names
    src_meta = convention_map.get(src_lower, src_lower)
    dst_meta = convention_map.get(dst_lower, dst_lower)
    
    # Try to use convert_eulers with metadata if both are recognized formats
    if src_meta in ('relion', 'dynamo') and dst_meta in ('relion', 'dynamo'):
        out = convert_eulers(arr, source_meta=src_meta, target_meta=dst_meta)
    else:
        # For other conventions, fall back to angles2matrix and matrix2angles
        from eulerangles import angles2matrix, matrix2angles
        mat = angles2matrix(
            arr,
            axes=src_convention.upper(),
            intrinsic=intrinsic,
            degrees=degrees
        )
        out = matrix2angles(
            mat,
            axes=dst_convention.upper(),
            intrinsic=intrinsic,
            degrees=degrees
        )
    
    if single:
        return out.reshape(3,)
    return out


def relion_to_dynamo_angles(relion_angles, degrees=True):
    """
    Convert RELION ZYZ (rlnAngleRot, Tilt, Psi) -> Dynamo ZXZ (tdrot, tilt, narot).
    
    This is a convenience function that uses convert_eulers with RELION and Dynamo metadata.
    
    Parameters
    ----------
    relion_angles : array-like
        Shape (N, 3) or (3,) RELION ZYZ Euler angles.
    degrees : bool
        Interpret input and output as degrees when True.
        Note: eulerangles convert_eulers always works in degrees.
    
    Returns
    -------
    np.ndarray
        Dynamo ZXZ Euler angles with same shape as input.
    """
    arr = np.asarray(relion_angles, dtype=float)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
        single = True
    
    out = convert_eulers(arr, source_meta='relion', target_meta='dynamo')
    
    if single:
        return out.reshape(3,)
    return out


def dynamo_to_relion_angles(dynamo_angles, degrees=True):
    """
    Convert Dynamo ZXZ (tdrot, tilt, narot) -> RELION ZYZ (rlnAngleRot, Tilt, Psi).
    
    This is a convenience function that uses convert_eulers with Dynamo and RELION metadata.
    
    Parameters
    ----------
    dynamo_angles : array-like
        Shape (N, 3) or (3,) Dynamo ZXZ Euler angles.
    degrees : bool
        Interpret input and output as degrees when True.
        Note: eulerangles convert_eulers always works in degrees.
    
    Returns
    -------
    np.ndarray
        RELION ZYZ Euler angles with same shape as input.
    """
    arr = np.asarray(dynamo_angles, dtype=float)
    single = False
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
        single = True
    
    out = convert_eulers(arr, source_meta='dynamo', target_meta='relion')
    
    if single:
        return out.reshape(3,)
    return out
