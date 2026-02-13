#!/usr/bin/env python3
"""
Convert RELION/Pytom STAR file to Dynamo TBL format.

This script converts particle STAR files to Dynamo .tbl format,
reorganizes the VLL file by tomogram names, and optionally adds
rlnTomoName column to the STAR file for ChimeraX visualization.

Usage:
    tomopanda-pick star2tbl --config config.yaml
"""

import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.io_dynamo import (
    relion_star_to_dynamo_tbl,
    read_vll_to_df,
    save_sorted_vll_by_tomonames,
)
from utils.tbl2star import add_rln_tomo_name_to_star
import starfile


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_star2tbl(config):
    """
    Execute star2tbl conversion from config dict.

    Expected config structure:
        input:
            starfile_particle: path to RELION/Pytom star file
            vll_path: path to vll file (tomogram paths for Dynamo)
        parameters:
            pixel_size: float (Angstrom)
            tomogram_size: [x, y, z] in pixels
        output:
            output_tbl: path for Dynamo .tbl file
            output_vll: path for sorted .vll file
            output_star_with_tomo_name: path for star with rlnTomoName (optional)
    """
    inp = config.get("input", {})
    params = config.get("parameters", {})
    out = config.get("output", {})

    starfile_particle = inp["starfile_particle"]
    vll_path = inp["vll_path"]
    pixel_size = params["pixel_size"]
    tomogram_size = tuple(params["tomogram_size"])
    output_tbl = out["output_tbl"]
    output_vll = out["output_vll"]
    output_star_with_tomo_name = out.get("output_star_with_tomo_name")

    # Step 1: Convert RELION star to Dynamo tbl
    print("Converting STAR to TBL: {} -> {}".format(starfile_particle, output_tbl))
    df = relion_star_to_dynamo_tbl(
        starfile_particle,
        pixel_size,
        tomogram_size=tomogram_size,
        output_file=output_tbl,
    )

    # Step 2: Extract tomogram names and reorganize vll file
    print("Reorganizing VLL file: {} -> {}".format(vll_path, output_vll))
    df_star = starfile.read(starfile_particle)
    tomo_names = (
        df_star["rlnTomoName"].unique()
        if "rlnTomoName" in df_star.columns
        else df_star["rlnMicrographName"].unique()
    )
    vll_df = read_vll_to_df(vll_path)
    save_sorted_vll_by_tomonames(tomo_names, vll_df, output_vll)

    # Step 3: Add rlnTomoName column to STAR (optional)
    if output_star_with_tomo_name:
        print("Adding rlnTomoName to STAR: -> {}".format(output_star_with_tomo_name))
        add_rln_tomo_name_to_star(
            star_path=starfile_particle,
            output_file=output_star_with_tomo_name,
            use_micrograph_name=True,
        )

    print("star2tbl conversion completed successfully.")


def main():
    """Main entry point for star2tbl command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert RELION/Pytom STAR file to Dynamo TBL format"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_star2tbl(config)


if __name__ == "__main__":
    main()
