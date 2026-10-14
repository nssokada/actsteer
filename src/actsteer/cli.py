"""Command-line entry point for stable ActSteer workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from actsteer import __version__
from actsteer.core.config import ActSteerConfig
from actsteer.core.datasets import ConceptDataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="actsteer", description=__doc__)
    parser.add_argument("--version", action="version", version=f"actsteer {__version__}")

    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config", help="Configuration utilities.")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="Validate a configuration file.")
    validate.add_argument("path", type=Path)

    data = commands.add_parser("data", help="Dataset ingestion utilities.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    validate_data = data_commands.add_parser(
        "validate", help="Validate normalized ConceptExample JSONL."
    )
    validate_data.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "config" and args.config_command == "validate":
        config = ActSteerConfig.from_yaml(args.path)
        print(f"valid: {args.path}")
        print(config.model_dump_json(indent=2))
        return 0
    if args.command == "data" and args.data_command == "validate":
        dataset = ConceptDataset.from_jsonl(args.path)
        summary = dataset.summary
        print(f"valid: {args.path}")
        print(f"examples: {summary.examples}")
        print(f"concepts: {summary.concepts}")
        print(f"references: {summary.references}")
        print(f"content_modes: {summary.content_modes}")
        print(f"fingerprint: {summary.fingerprint}")
        return 0
    raise AssertionError("unreachable command dispatch")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
