#!/usr/bin/env python3
"""
OncoLearn CLI - Main Entry Point

A comprehensive toolkit for cancer genomics analysis and biomarker discovery.
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="oncolearn",
        description="OncoLearn: A comprehensive toolkit for cancer genomics analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        help="Command to run"
    )

    # Import and register subcommands
    from . import download
    download.register_subcommand(subparsers)

    from . import preprocess
    preprocess.register_subcommand(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
