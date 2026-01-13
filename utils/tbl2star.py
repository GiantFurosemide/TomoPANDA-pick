"""
Utility functions for converting between Dynamo .tbl/.vll and RELION .star formats,
and adding rlnTomoName column to star files.

This module provides convenient wrapper functions for common conversion tasks.
"""

import starfile
from .io_dynamo import dynamo_tbl_vll_to_relion_star, relion_star_to_dynamo_tbl


def tbl_to_star(tbl_path, vll_path, output_file, pixel_size, tomogram_size=None, output_centered=True):
    """
    Convert Dynamo .tbl and .vll files to RELION .star format.
    
    Parameters
    ----------
    tbl_path : str
        Path to input Dynamo .tbl file.
    vll_path : str
        Path to input Dynamo .vll file (tomogram list).
    output_file : str
        Path to output RELION .star file.
    pixel_size : float
        Pixel size in Angstrom.
    tomogram_size : tuple of int, optional
        Tomogram size as (x, y, z) in pixels. If None, coordinates are assumed to be absolute.
    output_centered : bool, optional
        If True, output centered coordinates (rlnCenteredCoordinateXAngst, etc.).
        Default is True.
    
    Returns
    -------
    None
        The converted star file is written to output_file.
    
    Examples
    --------
    >>> tbl_to_star(
    ...     tbl_path="input.tbl",
    ...     vll_path="input.vll",
    ...     output_file="output.star",
    ...     pixel_size=6.72,
    ...     tomogram_size=(999, 999, 499),
    ...     output_centered=True
    ... )
    """
    dynamo_tbl_vll_to_relion_star(
        tbl_path=tbl_path,
        vll_path=vll_path,
        output_file=output_file,
        pixel_size=pixel_size,
        tomogram_size=tomogram_size,
        output_centered=output_centered
    )


def star_to_tbl(star_path, pixel_size, tomogram_size=None, output_file=None):
    """
    Convert RELION .star file to Dynamo .tbl format.
    
    Parameters
    ----------
    star_path : str
        Path to input RELION .star file.
    pixel_size : float
        Pixel size in Angstrom.
    tomogram_size : tuple of int, optional
        Tomogram size as (x, y, z) in pixels. If None, coordinates are assumed to be absolute.
    output_file : str, optional
        Path to output Dynamo .tbl file. If None, will be generated from star_path.
        Default is None.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the converted Dynamo table data.
    
    Examples
    --------
    >>> df = star_to_tbl(
    ...     star_path="input.star",
    ...     pixel_size=6.72,
    ...     tomogram_size=(999, 999, 499),
    ...     output_file="output.tbl"
    ... )
    """
    if output_file is None:
        output_file = star_path.replace('.star', '.tbl')
    
    df = relion_star_to_dynamo_tbl(
        star_path=star_path,
        pixel_size=pixel_size,
        tomogram_size=tomogram_size,
        output_file=output_file
    )
    return df


def add_rln_tomo_name_to_star(star_path, output_file=None, use_micrograph_name=True):
    """
    Add rlnTomoName column to a RELION .star file.
    
    The rlnTomoName column is created by mapping unique micrograph names to sequential
    integer IDs starting from 1.
    
    Parameters
    ----------
    star_path : str
        Path to input RELION .star file.
    output_file : str, optional
        Path to output .star file. If None, will append '_tomo_name' before the .star extension.
        Default is None.
    use_micrograph_name : bool, optional
        If True, use 'rlnMicrographName' column to create rlnTomoName mapping.
        If False, use 'rlnTomoName' column if it exists. Default is True.
    
    Returns
    -------
    str
        Path to the output file that was written.
    
    Examples
    --------
    >>> output_path = add_rln_tomo_name_to_star(
    ...     star_path="input.star",
    ...     output_file="output_tomo_name.star"
    ... )
    """
    # Read the star file
    rl_particles = starfile.read(star_path)
    
    # Check which columns exist
    has_micrograph_name = 'rlnMicrographName' in rl_particles.columns
    has_tomo_name = 'rlnTomoName' in rl_particles.columns
    
    # Case 1: Has rlnMicrographName but no rlnTomoName
    # Execute original logic: create rlnTomoName from rlnMicrographName
    if has_micrograph_name and not has_tomo_name:
        source_col = 'rlnMicrographName'
        unique_names = rl_particles[source_col].unique()
        name_to_tomo_number = {name: idx + 1 for idx, name in enumerate(unique_names)}
        rl_particles['rlnTomoName'] = rl_particles[source_col].map(name_to_tomo_number)
    
    # Case 2: No rlnMicrographName but has rlnTomoName
    # Create rlnMicrographName from rlnTomoName, then process rlnTomoName
    elif not has_micrograph_name and has_tomo_name:
        # Copy rlnTomoName values to rlnMicrographName
        rl_particles['rlnMicrographName'] = rl_particles['rlnTomoName']
        # Process rlnTomoName: create mapping from unique values
        source_col = 'rlnTomoName'
        unique_names = rl_particles[source_col].unique()
        name_to_tomo_number = {name: idx + 1 for idx, name in enumerate(unique_names)}
        rl_particles['rlnTomoName'] = rl_particles[source_col].map(name_to_tomo_number)
    
    # Case 3: Both columns exist
    # No processing needed, output as is
    elif has_micrograph_name and has_tomo_name:
        pass  # Do nothing, output unchanged
    
    # Case 4: Neither column exists
    # Raise error
    else:
        raise ValueError("Neither 'rlnMicrographName' nor 'rlnTomoName' column found in star file")
    
    # Determine output file path
    if output_file is None:
        output_file = star_path.replace('.star', '_tomo_name.star')
    
    # Write the updated star file
    starfile.write(rl_particles, output_file)
    
    return output_file
