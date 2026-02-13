#!/usr/bin/env python3
"""
Convert Dynamo TBL/VLL files to RELION STAR format.

This script converts Dynamo .tbl and .vll files to RELION .star format,
and optionally adds rlnTomoName column for ChimeraX visualization.

Usage:
    tomopanda-pick tbl2star --config config.yaml
"""

import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.tbl2star import tbl_to_star, add_rln_tomo_name_to_star


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_tbl2star(config):
    """
    Execute tbl2star conversion from config dict.

    Expected config structure:
        input:
            input_tbl: path to Dynamo .tbl file
            input_vll: path to Dynamo .vll file
        parameters:
            pixel_size: float (Angstrom)
            tomogram_size: [x, y, z] in pixels
            output_centered: bool (default True)
        output:
            output_star: path for RELION .star file
            output_star_with_tomo_name: path for star with rlnTomoName (optional)
    """
    inp = config.get("input", {})
    params = config.get("parameters", {})
    out = config.get("output", {})

    input_tbl = inp["input_tbl"]
    input_vll = inp["input_vll"]
    pixel_size = params["pixel_size"]
    tomogram_size = tuple(params.get("tomogram_size", [])) if params.get("tomogram_size") else None
    output_centered = params.get("output_centered", True)
    output_star = out["output_star"]
    output_star_with_tomo_name = out.get("output_star_with_tomo_name")

    # Step 1: Convert Dynamo tbl/vll to RELION star
    print("Converting TBL/VLL to STAR: {} + {} -> {}".format(input_tbl, input_vll, output_star))
    tbl_to_star(
        tbl_path=input_tbl,
        vll_path=input_vll,
        output_file=output_star,
        pixel_size=pixel_size,
        tomogram_size=tomogram_size,
        output_centered=output_centered,
    )

    # Step 2: Add rlnTomoName column (optional)
    if output_star_with_tomo_name:
        print("Adding rlnTomoName to STAR: -> {}".format(output_star_with_tomo_name))
        add_rln_tomo_name_to_star(
            star_path=output_star,
            output_file=output_star_with_tomo_name,
            use_micrograph_name=True,
        )

    print("tbl2star conversion completed successfully.")


def main():
    """Main entry point for tbl2star command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Dynamo TBL/VLL files to RELION STAR format"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_tbl2star(config)


if __name__ == "__main__":
    main()
