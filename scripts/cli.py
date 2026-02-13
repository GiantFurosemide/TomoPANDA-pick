#!/usr/bin/env python3
"""
Main CLI entry point for TomoPANDA-pick.

Usage:
    tomopanda-pick star2tbl --config a.yaml
    tomopanda-pick tbl2star --config b.yaml
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="tomopanda-pick",
        description="TomoPANDA-pick: 3D particle picking and format conversion tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # star2tbl subcommand
    star2tbl_parser = subparsers.add_parser("star2tbl", help="Convert STAR file to Dynamo TBL format")
    star2tbl_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    star2tbl_parser.set_defaults(func="star2tbl")

    # tbl2star subcommand
    tbl2star_parser = subparsers.add_parser("tbl2star", help="Convert Dynamo TBL/VLL to RELION STAR format")
    tbl2star_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    tbl2star_parser.set_defaults(func="tbl2star")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "star2tbl":
        from scripts.star2tbl import load_config, run_star2tbl
        config = load_config(args.config)
        run_star2tbl(config)
    elif args.command == "tbl2star":
        from scripts.tbl2star import load_config, run_tbl2star
        config = load_config(args.config)
        run_tbl2star(config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
