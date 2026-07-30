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
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .corpus import load_corpus
from .models import AnthropicModelClient
from .models import THINKING_OFF_BY_DEFAULT, default_thinks
from .report import (
    build_report, build_source_scores, cost_summary, judge_failure_note,
    regime_note, source_scores_markdown, to_markdown, truncation_note,
)
from .runner import CONTROL, load_completed, regrade_rows, run_control, run_corpus


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


def _assemble_report(rows, *, strong: str, weak: str):
    """The full markdown report for `rows`, and the arena subset it ranked.

    Shared by `run` and `regrade` so a re-graded report is the same document as
    a freshly-run one — a regrade that formatted differently would invite
    comparing it against the wrong baseline."""
    # Keep the control arm out of the ranked scoreboard.
    arena_rows = [r for r in rows if r.conversion != CONTROL]
    report = build_report(arena_rows, strong=strong, weak=weak)
    md = to_markdown(report)
    source_md = source_scores_markdown(build_source_scores(arena_rows), report.models)
    if source_md:
        md += "\n\n" + source_md
    trunc_md = truncation_note(arena_rows)
    if trunc_md:
        md += "\n\n" + trunc_md
    # A judge that failed produced a 0 that is not about the representation,
    # and those land preferentially on the weaker arms — flag before the cost.
    judge_md = judge_failure_note(arena_rows)
    if judge_md:
        md += "\n\n" + judge_md
    # Cost covers every row the run billed, control arm included.
    cost_md = cost_summary(rows)
    if cost_md:
        md += "\n\n" + cost_md
    # Regime last and over ALL rows: it qualifies every number above it, and a
    # report that cannot say how it was measured should say so on its face.
    regime_md = regime_note(rows)
    if regime_md:
        md += "\n\n" + regime_md
    return md, arena_rows


def _check_models(parser, models: dict, allow_thinking: bool) -> None:
    """Refuse to bill a run against a model whose reasoning regime isn't the one
    every published number was measured under.

    `models` maps a flag name to the model it names. The harness sends no
    `thinking` parameter (models.py), which means thinking-off ONLY on models
    whose default is off. On Opus 5 / Sonnet 5 / Fable 5 the same omitted
    parameter runs adaptive thinking, which breaks the arena three ways at
    once: a strong reader that reasons around a corrupted conversion is the
    confound the design exists to exclude, thinking depth follows the input so
    the arms think different amounts, and thinking is charged against
    --max-tokens alongside the answer.

    An unclassified model is refused too. The alternative is assuming a default
    for a model nobody checked, which is precisely how this became a problem:
    the assumption was true for years, stayed written down as though it were a
    property of the request, and then quietly stopped being true.

    This is a startup error rather than a footnote because the failure is
    otherwise silent -- no exception, no empty output, just plausible numbers
    from a different experiment landing in the same JSONL as the old ones."""
    for flag, model in models.items():
        if not model:
            continue
        thinks = default_thinks(model)
        if thinks is False:
            continue
        if thinks is True and allow_thinking:
            continue
        if thinks is True:
            parser.error(
                f"{flag} {model!r} runs adaptive thinking when `thinking` is "
                f"omitted, which this harness relies on NOT happening -- every "
                f"baseline on disk was measured thinking-off, so its numbers "
                f"would not be comparable. Use one of: "
                f"{', '.join(sorted(THINKING_OFF_BY_DEFAULT))}. "
                f"Pass --allow-thinking-default to run it anyway as a deliberate "
                f"experiment (the report will flag every affected row)."
            )
        parser.error(
            f"{flag} {model!r} has no thinking-default on file, so whether it "
            f"would think is unknown and the run's regime is undefined. Add it "
            f"to THINKING_OFF_BY_DEFAULT or THINKING_ON_BY_DEFAULT in "
            f"decant_eval/models.py once verified against the model docs."
        )


def _regrade(args) -> int:
    """Re-grade an audit trail against the current graders and questions."""
    cases = load_corpus(args.corpus, split=args.split)
    rows, _ = load_completed(args.rows)
    if not rows:
        print(f"{args.rows}: no rows to re-grade")
        return 1

    judge = AnthropicModelClient() if args.judge else None
    new_rows, changed, skipped = regrade_rows(
        rows, cases, judge=judge, judge_model=args.judge or "")

    out_rows = args.out_rows or f"{args.out}.jsonl"
    with open(out_rows, "w", encoding="utf-8") as fh:
        for r in new_rows:
            fh.write(json.dumps(asdict(r), ensure_ascii=True) + "\n")

    md, arena_rows = _assemble_report(new_rows, strong=args.strong, weak=args.weak)
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"\nRe-graded {len(rows)} row(s): {changed} verdict(s) changed.")
    if skipped:
        print(f"{skipped} row(s) kept their stored verdict — `exact`/`open` need a "
              f"judge; pass --judge MODEL to re-grade them too (BILLED).")
    print(f"Wrote {args.out} ({len(arena_rows)} arena rows) and {out_rows}; "
          f"{args.rows} unchanged.")
    return 0


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
    run.add_argument(
        "--split", default="all", choices=("all", "dev", "test", "retired"),
        help="question bank to score (default all = everything not retired). "
             "Iterate compression candidates on `dev` and keep `test` held out "
             "until declaring one done — tuning against the questions you also "
             "report on converges on keeping just the asked-about facts.",
    )
    run.add_argument(
        "--weak-only", action="store_true",
        help="run the weak target model alone. Weak-reader accuracy is the "
             "binding constraint, so loop iterations don't need the strong tier; "
             "this cuts a lever test to a fraction of the cost. No reliability "
             "spread is reported (it needs both tiers).",
    )
    run.add_argument(
        "--allow-thinking-default", action="store_true",
        help="run a model that thinks by default (Opus 5, Sonnet 5, Fable 5). "
             "Refused otherwise: the harness sends no `thinking` parameter and "
             "reads that as thinking-off, which is only true on the older tiers, "
             "and every baseline on disk was measured that way. Thinking also "
             "bills as output tokens and eats --max-tokens. The report flags "
             "every affected row regardless of this flag.",
    )
    run.add_argument(
        "--repeats", type=int, default=1, metavar="N",
        help="ask each question N times and average (default 1). Sampling can't "
             "be pinned on the strong tier, so a single sample per cell is one "
             "draw, not a measurement — repeats are the only variance control. "
             "Cost scales linearly with N.",
    )

    rg = sub.add_parser(
        "regrade",
        help="re-grade an existing JSONL audit trail and rewrite the report (free by default)",
    )
    rg.add_argument("--corpus", required=True, help="corpus directory (supplies the questions)")
    rg.add_argument(
        "--split", default="all", choices=("all", "dev", "test", "retired"),
        help="question bank to re-grade (default all = everything not retired). "
             "Rows whose question the filter excludes keep their stored verdict.",
    )
    rg.add_argument("--rows", required=True, help="JSONL audit trail to re-grade")
    rg.add_argument("--out", default="report-regraded.md", help="output markdown path")
    rg.add_argument(
        "--out-rows", default=None,
        help="re-graded JSONL (default <out>.jsonl). The input file is never "
             "modified — it is the record of what the run actually returned.",
    )
    rg.add_argument("--strong", default="claude-opus-4-8")
    rg.add_argument("--weak", default="claude-haiku-4-5")
    rg.add_argument(
        "--judge", default=None, metavar="MODEL",
        help="BILLED. Also re-grade `exact` and `open` rows, which need a judge. "
             "Omitted by default so a regrade costs nothing; those rows then keep "
             "their stored verdict.",
    )

    args = parser.parse_args(argv)
    if args.cmd == "regrade":
        # Only --judge is billed here; --strong/--weak are labels for the
        # scoreboard, and validating them would block re-grading a file that
        # already contains rows from a model we refuse to run.
        _check_models(parser, {"--judge": args.judge}, allow_thinking=False)
        return _regrade(args)
    if args.cmd != "run":  # pragma: no cover - argparse enforces
        parser.error("unknown command")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    _check_models(
        parser,
        {"--strong": None if args.weak_only else args.strong,
         "--weak": args.weak, "--judge": args.judge},
        allow_thinking=args.allow_thinking_default,
    )

    cases = load_corpus(args.corpus, split=args.split)
    client = AnthropicModelClient()
    models = [args.weak] if args.weak_only else [args.strong, args.weak]
    rows_path = args.rows or f"{args.out}.jsonl"

    rows = run_corpus(
        cases,
        client=client,
        models=models,
        judge=client,
        judge_model=args.judge,
        max_tokens=args.max_tokens,
        repeats=args.repeats,
        raw_arena=not args.no_raw,
        jsonl_path=rows_path,
        resume=args.resume,
    )
    md, arena_rows = _assemble_report(rows, strong=args.strong, weak=args.weak)

    if not args.no_control:
        control_rows = []
        for case in cases:
            control_rows.extend(
                run_control(case, client=client, models=models,
                            judge=client, judge_model=args.judge,
                            max_tokens=args.max_tokens)
            )
        md += "\n" + _control_note(control_rows)

    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {args.out} ({len(arena_rows)} rows across {len(cases)} case(s)); "
          f"audit trail {rows_path}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
