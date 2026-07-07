"""Command-line entry: run the arena against a real corpus and write a report.

    python -m decant_eval.cli run --corpus ./corpus \
        --strong claude-opus-4-8 --weak claude-haiku-4-5 --out report.md

Needs Anthropic credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).
Offline development and the test suite use FakeModelClient instead — this entry
is the one place the real SDK is touched.

Rows stream to a JSONL sidecar (``--rows``, default ``<out>.jsonl``) as they
complete, so a crash mid-run loses nothing; ``--resume`` continues from it. The
memory-contamination control arm (``--control``, on by default) runs each
question with no document and flags any answered from the model's memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .corpus import load_corpus
from .models import AnthropicModelClient
from .report import build_report, to_markdown
from .runner import CONTROL, run_control, run_corpus


def _control_note(control_rows) -> str:
    """A short section flagging questions answered from memory (no document)."""
    memorized = [r for r in control_rows if r.correct]
    lines = ["", "## Memory-contamination control", ""]
    if not memorized:
        lines.append(
            "_No question was answered correctly with no document — the arena scores "
            "reflect the representation, not the model's prior knowledge._"
        )
    else:
        lines.append(
            f"_{len(memorized)} of {len(control_rows)} question x model runs were answered "
            "correctly with NO document -- those answers come from training data, so a "
            "conversion 'transferring' them proves nothing. Discount or exclude them:_"
        )
        lines.append("")
        for r in memorized:
            lines.append(f"- `{r.case}` / `{r.question_id}` ({r.model})")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="decant_eval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="score a corpus and write a report")
    run.add_argument("--corpus", required=True, help="corpus directory")
    run.add_argument("--strong", default="claude-opus-4-8", help="strong target model")
    run.add_argument("--weak", default="claude-haiku-4-5", help="weak target model")
    run.add_argument("--judge", default="claude-opus-4-8", help="judge model for open questions")
    run.add_argument("--out", default="report.md", help="output markdown path")
    run.add_argument("--rows", default=None, help="JSONL audit trail (default <out>.jsonl)")
    run.add_argument("--resume", action="store_true", help="continue from an existing --rows file")
    run.add_argument("--no-raw", action="store_true", help="skip the source-PDF raw-upload baseline")
    run.add_argument("--no-control", action="store_true", help="skip the no-document control arm")
    run.add_argument("--max-tokens", type=int, default=512)

    args = parser.parse_args(argv)
    if args.cmd != "run":  # pragma: no cover - argparse enforces
        parser.error("unknown command")

    cases = load_corpus(args.corpus)
    client = AnthropicModelClient()
    models = [args.strong, args.weak]
    rows_path = args.rows or f"{args.out}.jsonl"

    rows = run_corpus(
        cases,
        client=client,
        models=models,
        judge=client,
        judge_model=args.judge,
        max_tokens=args.max_tokens,
        raw_arena=not args.no_raw,
        jsonl_path=rows_path,
        resume=args.resume,
    )
    # Keep the control arm out of the ranked scoreboard.
    arena_rows = [r for r in rows if r.conversion != CONTROL]
    report = build_report(arena_rows, strong=args.strong, weak=args.weak)
    md = to_markdown(report)

    if not args.no_control:
        control_rows = []
        for case in cases:
            control_rows.extend(
                run_control(case, client=client, models=models,
                            judge=client, judge_model=args.judge, max_tokens=args.max_tokens)
            )
        md += "\n" + _control_note(control_rows)

    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {args.out} ({len(arena_rows)} rows across {len(cases)} case(s)); "
          f"audit trail {rows_path}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
