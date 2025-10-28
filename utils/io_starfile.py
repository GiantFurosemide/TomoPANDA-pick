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


