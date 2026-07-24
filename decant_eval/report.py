"""Aggregate run rows into the scoreboard.

Two axes per conversion, the whole point of the harness:
  - accuracy  — mean question score, per target-model tier
  - cost      — mean input tokens per question (the conversion's token weight)

and the novel metric:
  - reliability spread = strong_accuracy − weak_accuracy. A conversion that lets
    the weak model answer as well as the strong one has a *small* spread — it
    transfers meaning robustly, not just to a model smart enough to recover from
    a bad representation. Lower is better.

Metric-integrity guards (a scoreboard that misleads is worse than none):
  - Conversions are compared only on the cases where *all* of them appear, so
    two conversions aren't ranked on different question subsets. Excluded cases
    are reported; if the conversions share no common case the report says so
    rather than pretending the numbers are comparable.
  - A conversion missing the strong tier ranks *last*, not by a mean of whatever
    tiers it does have — a partial run must not out-rank a complete one.
  - Spread is annotated, not just printed: two tiers that fail a conversion
    equally (0 vs 0) yield spread 0.00, which reads "robust" but means
    "uniformly useless". Below an accuracy floor the spread is flagged.
  - Cost is per-model. Opus and Haiku tokenize differently, so averaging their
    input-token counts into one number compares apples to oranges; the strong
    tier's cost is the reported figure.
  - A failed model call (Result.status set — e.g. the document exceeds the
    target model's context window) is scored 0 by the runner but must not read
    as "answered everything wrong": affected cells are flagged with `!` and a
    footnote counts the failed calls per arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from .runner import CONTEXT_OVERFLOW

# Below this strong-tier accuracy, a small spread means "uniformly useless", not
# "robust" — so the spread is flagged rather than read as a virtue.
SPREAD_ACCURACY_FLOOR = 0.25

# Footnote wording for Result.status values; an unrecognized status falls back
# to the raw value with underscores spaced.
_FAILURE_REASONS = {
    CONTEXT_OVERFLOW: "document does not fit the model's context window",
}


@dataclass
class ConversionScore:
    conversion: str
    accuracy: dict[str, float] = field(default_factory=dict)  # model -> mean score
    cost_by_model: dict[str, float] = field(default_factory=dict)  # model -> mean input tok
    spread: float | None = None  # strong_acc - weak_acc, when both tiers ran
    spread_reliable: bool = False  # False when strong accuracy is below the floor
    n_cases: int = 0  # cases contributing (after the common-case restriction)
    # model -> (failed, total, statuses) over this conversion's scored rows,
    # present only when failed > 0: calls that failed outright (Result.status
    # set, e.g. context overflow) and were scored 0 without an answer.
    failed_calls: dict[str, tuple[int, int, tuple[str, ...]]] = field(default_factory=dict)


@dataclass
class Report:
    scores: list[ConversionScore]
    strong: str | None
    weak: str | None
    models: list[str]
    common_cases: list[str]
    excluded_cases: list[str]
    comparable: bool  # False when conversions share no common case


def build_report(
    rows, *, strong: str | None = None, weak: str | None = None,
    floor: float = SPREAD_ACCURACY_FLOOR,
) -> Report:
    models = sorted({r.model for r in rows})
    conversions = sorted({r.conversion for r in rows})
    all_cases = sorted({r.case for r in rows})

    # Compare on cases where every conversion appears, so no two conversions are
    # ranked on different question subsets.
    cases_by_conv = {
        conv: {r.case for r in rows if r.conversion == conv} for conv in conversions
    }
    common = set(all_cases)
    for cases in cases_by_conv.values():
        common &= cases
    comparable = bool(common)
    scoring_cases = common if comparable else set(all_cases)
    excluded = sorted(set(all_cases) - scoring_cases)

    scores: list[ConversionScore] = []
    for conv in conversions:
        cs = ConversionScore(conversion=conv)
        conv_rows = [r for r in rows if r.conversion == conv and r.case in scoring_cases]
        cs.n_cases = len({r.case for r in conv_rows})
        for model in models:
            mr = [r for r in conv_rows if r.model == model]
            if mr:
                cs.accuracy[model] = mean(r.score for r in mr)
                cs.cost_by_model[model] = mean(r.input_tokens for r in mr)
                failed = [r for r in mr if getattr(r, "status", "")]
                if failed:
                    statuses = tuple(sorted({r.status for r in failed}))
                    cs.failed_calls[model] = (len(failed), len(mr), statuses)
        if strong in cs.accuracy and weak in cs.accuracy:
            cs.spread = cs.accuracy[strong] - cs.accuracy[weak]
            cs.spread_reliable = cs.accuracy[strong] >= floor
        scores.append(cs)

    # Rank: strong-model accuracy (missing strong tier sinks to the bottom, not
    # substituted by a mean), then cheapest on the strong tier, then — only when
    # the spread is reliable — tightest spread. None spread sorts last.
    def key(cs: ConversionScore):
        has_strong = strong in cs.accuracy
        acc = cs.accuracy.get(strong, 0.0)
        cost = cs.cost_by_model.get(strong, float("inf"))
        spread = cs.spread if (cs.spread is not None and cs.spread_reliable) else float("inf")
        return (0 if has_strong else 1, -acc, cost, spread)

    scores.sort(key=key)
    return Report(
        scores=scores, strong=strong, weak=weak, models=models,
        common_cases=sorted(scoring_cases), excluded_cases=excluded, comparable=comparable,
    )


@dataclass
class SourceScore:
    """Accuracy of one conversion on the questions tagged with one answer
    location (Question.source) within one case."""
    case: str
    source: str
    conversion: str
    accuracy: dict[str, float] = field(default_factory=dict)  # model -> mean score
    n_questions: int = 0


def build_source_scores(rows) -> list[SourceScore]:
    """Per-(case, source, conversion) accuracy over the rows whose question
    carries a source tag. Empty when nothing is tagged. Source tags are
    case-scoped ("figure-12" in one case is unrelated to another's), so slices
    are never aggregated across cases; within a (case, source) group every
    conversion answered the same questions, so the cells compare directly —
    e.g. decant vs decant-plain on figure-tagged questions is the companion
    PDF's per-figure contribution."""
    tagged = [r for r in rows if getattr(r, "source", "")]
    out: list[SourceScore] = []
    groups = sorted({(r.case, r.source, r.conversion) for r in tagged})
    for case, source, conv in groups:
        grp = [r for r in tagged if (r.case, r.source, r.conversion) == (case, source, conv)]
        ss = SourceScore(case=case, source=source, conversion=conv)
        ss.n_questions = len({r.question_id for r in grp})
        for model in sorted({r.model for r in grp}):
            ss.accuracy[model] = mean(r.score for r in grp if r.model == model)
        out.append(ss)
    return out


def source_scores_markdown(scores: list[SourceScore], models: list[str]) -> str:
    """A '## By answer source' appendix table, or "" when nothing is tagged."""
    if not scores:
        return ""
    lines = ["## By answer source", ""]
    header = ["case", "source", "conversion", *models, "n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for ss in scores:
        cells = [ss.case, ss.source, ss.conversion]
        for m in models:
            cells.append(f"{ss.accuracy[m]:.2f}" if m in ss.accuracy else "-")
        cells.append(str(ss.n_questions))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "_Accuracy sliced by where the answer lives in the source (the questions' "
        "`source` tags). Tags are case-scoped; conversions within one case+source "
        "group answered the same questions and compare directly._"
    )
    return "\n".join(lines)


def to_markdown(report: Report) -> str:
    lines = ["# Decant eval report", ""]
    header = ["conversion", *report.models, "cost (strong tok)", "spread", "n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for cs in report.scores:
        cells = [cs.conversion]
        for m in report.models:
            # `!` — the mean includes calls that failed outright; see footnote.
            flag = "!" if m in cs.failed_calls else ""
            cells.append(f"{cs.accuracy[m]:.2f}{flag}" if m in cs.accuracy else "-")
        strong_cost = cs.cost_by_model.get(report.strong)
        cost_flag = "!" if report.strong in cs.failed_calls else ""
        cells.append(f"{strong_cost:.0f}{cost_flag}" if strong_cost is not None else "-")
        # Spread built on a tier with failed calls inherits the flag — a +0.70
        # spread from a document that never fit the weak reader must not read
        # as "the weak model answered wrong".
        spread_flag = "!" if (
            report.strong in cs.failed_calls or report.weak in cs.failed_calls
        ) else ""
        if cs.spread is None:
            cells.append("-")
        elif cs.spread_reliable:
            cells.append(f"{cs.spread:+.2f}{spread_flag}")
        else:
            cells.append(f"{cs.spread:+.2f}*{spread_flag}")  # below the accuracy floor — see note
        cells.append(str(cs.n_cases))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if report.strong and report.weak:
        lines.append(
            f"_Spread = {report.strong} accuracy - {report.weak} accuracy; "
            "lower means the conversion transfers meaning robustly to the weaker reader. "
            f"Cost is {report.strong} input tokens (tiers tokenize differently)._"
        )
    if any(cs.spread is not None and not cs.spread_reliable for cs in report.scores):
        lines.append(
            f"_* spread is below the {SPREAD_ACCURACY_FLOOR:.0%} accuracy floor -- "
            "a tight spread here means uniformly wrong, not robust._"
        )
    for cs in report.scores:
        for model in sorted(cs.failed_calls):
            n_failed, n_total, statuses = cs.failed_calls[model]
            reasons = "; ".join(_FAILURE_REASONS.get(s, s.replace("_", " ")) for s in statuses)
            lines.append(
                f"_! {cs.conversion} / {model}: {n_failed}/{n_total} calls failed "
                f"({reasons}); failed calls score 0 with 0 tokens billed -- the "
                "document was never read, not answered wrong._"
            )
    if not report.comparable:
        lines.append(
            "_WARNING: conversions share no common case; scores are over each conversion's "
            "own cases and are NOT directly comparable._"
        )
    elif report.excluded_cases:
        lines.append(
            "_Compared on "
            + f"{len(report.common_cases)} common case(s); excluded (not present for every "
            + f"conversion): {', '.join(report.excluded_cases)}._"
        )
    return "\n".join(lines)
