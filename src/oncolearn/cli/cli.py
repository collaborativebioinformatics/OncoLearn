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
    from oncolearn.cli.subcommands.train.command       import register_subcommand as register_train
    from oncolearn.cli.subcommands.tcia.command        import register_subcommand as register_tcia
    from oncolearn.cli.subcommands.xena.command        import register_subcommand as register_xena
    from oncolearn.cli.subcommands.cbioportal.command  import register_subcommand as register_cbioportal
    from oncolearn.cli.subcommands.preprocess.command  import register_subcommand as register_preprocess

    register_train(subparsers)
    register_tcia(subparsers)
    register_xena(subparsers)
    register_cbioportal(subparsers)
    register_preprocess(subparsers)

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
