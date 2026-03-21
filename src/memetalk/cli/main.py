from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from memetalk.app.container import build_container
from memetalk.config import AppSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memetalk")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index meme images")
    index_subparsers = index_parser.add_subparsers(dest="index_command", required=True)

    build_parser_cmd = index_subparsers.add_parser("build", help="Build or update the meme index")
    build_parser_cmd.add_argument("--source", required=True, help="Source directory containing images")
    build_parser_cmd.add_argument("--reindex", action="store_true", help="Force reindex even if the hash already exists")

    eval_parser = subparsers.add_parser("eval", help="Run offline search evaluation")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_run_parser = eval_subparsers.add_parser("run", help="Run evaluation cases")
    eval_run_parser.add_argument("--cases", required=True, help="JSON file containing evaluation cases")

    eval_tune_parser = eval_subparsers.add_parser("tune", help="Tune deterministic scoring weights")
    eval_tune_parser.add_argument("--cases", required=True, help="JSON file containing evaluation cases")
    eval_tune_parser.add_argument("--output", help="Path to write the tuned scoring profile JSON")
    eval_tune_parser.add_argument("--passes", type=int, default=2, help="Number of coordinate-search passes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index" and args.index_command == "build":
        settings = AppSettings.from_env()
        container = build_container(settings)
        summary = container.indexing_service.build_index(Path(args.source), reindex=args.reindex)
        print(json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval" and args.eval_command == "run":
        settings = AppSettings.from_env()
        container = build_container(settings)
        cases = container.evaluation_service.load_cases(Path(args.cases))
        report = container.evaluation_service.run_cases(cases)
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval" and args.eval_command == "tune":
        settings = AppSettings.from_env()
        container = build_container(settings)
        cases = container.evaluation_service.load_cases(Path(args.cases))
        tuning_report = container.evaluation_service.tune_scoring_profile(cases, passes=args.passes)
        output_path = Path(args.output) if args.output else settings.search_scoring_profile_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(tuning_report.best_profile.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    **tuning_report.model_dump(mode="json"),
                    "output_path": str(output_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
