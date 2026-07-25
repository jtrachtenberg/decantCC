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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

# Below this strong-tier accuracy, a small spread means "uniformly useless", not
# "robust" — so the spread is flagged rather than read as a virtue.
SPREAD_ACCURACY_FLOOR = 0.25


@dataclass
class ConversionScore:
    conversion: str
    accuracy: dict[str, float] = field(default_factory=dict)  # model -> mean score
    cost_by_model: dict[str, float] = field(default_factory=dict)  # model -> mean input tok
    spread: float | None = None  # strong_acc - weak_acc, when both tiers ran
    spread_reliable: bool = False  # False when strong accuracy is below the floor
    n_cases: int = 0  # cases contributing (after the common-case restriction)


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


# Per-million-token list prices, USD. Cached 2026-07-24 — verify against
# platform.claude.com/docs/en/pricing before quoting a figure that matters.
PRICES = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Cache multipliers on the base input price: a read is ~0.1x, a write ~1.25x at
# the 5-minute TTL the harness uses (models.py sets no explicit ttl).
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def billed_input_tokens(row) -> float:
    """`row`'s input priced in full-rate-equivalent tokens.

    input_tokens is the total the request covered; the cached portions inside it
    bill at a fraction of base rate, so charging all of it at full price
    overstates a cached run several fold. Rows written before the components
    were recorded carry 0/0 and therefore price as fully uncached — an upper
    bound, which cost_summary labels rather than passing off as measured."""
    read = getattr(row, "cache_read_tokens", 0) or 0
    written = getattr(row, "cache_creation_tokens", 0) or 0
    uncached = max(0, row.input_tokens - read - written)
    return uncached + read * CACHE_READ_MULTIPLIER + written * CACHE_WRITE_MULTIPLIER


def cost_summary(rows) -> str:
    """A '## Cost' table: per-model billed tokens and USD at list price."""
    models = sorted({r.model for r in rows})
    if not models:
        return ""
    lines = ["## Cost", "", "| model | billed input tok | output tok | USD (list) |",
             "| --- | --- | --- | --- |"]
    total = 0.0
    unpriced, uninstrumented = [], 0
    for m in models:
        mr = [r for r in rows if r.model == m]
        bin_tok = sum(billed_input_tokens(r) for r in mr)
        out_tok = sum(r.output_tokens for r in mr)
        uninstrumented += sum(
            1 for r in mr
            if r.input_tokens and not (getattr(r, "cache_read_tokens", 0)
                                       or getattr(r, "cache_creation_tokens", 0))
        )
        if m in PRICES:
            in_price, out_price = PRICES[m]
            usd = bin_tok / 1e6 * in_price + out_tok / 1e6 * out_price
            total += usd
            cell = f"${usd:,.2f}"
        else:
            unpriced.append(m)
            cell = "-"
        lines.append(f"| {m} | {bin_tok:,.0f} | {out_tok:,} | {cell} |")
    lines.append(f"| **total** | | | **${total:,.2f}** |")
    lines.append("")
    lines.append(
        "_Billed input counts cache reads at "
        f"{CACHE_READ_MULTIPLIER}x and writes at {CACHE_WRITE_MULTIPLIER}x base rate, "
        "so it is well below the raw input-token total on a cached run. List "
        "prices only — no discounts, batch rates, or negotiated terms._"
    )
    if unpriced:
        lines.append(f"_No price on file for: {', '.join(unpriced)} — see PRICES in report.py._")
    if uninstrumented:
        lines.append(
            f"_WARNING: {uninstrumented} row(s) predate cache-component recording and are "
            "priced as fully uncached. The real figure is lower; treat this as a ceiling._"
        )
    return "\n".join(lines)


def truncation_note(rows) -> str:
    """A warning listing rows the model was cut off on (stop_reason
    "max_tokens"), or "" when none were. Such a row scored what it scored
    because of the output budget, not because the conversion failed to carry
    the fact — reading it as the latter is exactly the misreading the
    scoreboard's other guards exist to prevent. Raise --max-tokens and re-run
    the affected rows (--resume skips the rest)."""
    hits = [r for r in rows if getattr(r, "truncated", False)]
    if not hits:
        return ""
    lines = [
        "## Truncated answers",
        "",
        f"_WARNING: {len(hits)} of {len(rows)} answers hit the output-token cap "
        "(stop_reason `max_tokens`) and were cut off mid-answer. Their scores "
        "measure the budget, not the representation -- re-run them with a higher "
        "`--max-tokens` before drawing conclusions:_",
        "",
    ]
    for r in hits:
        lines.append(f"- `{r.case}` / `{r.question_id}` ({r.model}, {r.conversion})")
    return "\n".join(lines)


def to_markdown(report: Report) -> str:
    lines = ["# Decant eval report", ""]
    header = ["conversion", *report.models, "cost (strong tok)", "spread", "n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for cs in report.scores:
        cells = [cs.conversion]
        for m in report.models:
            cells.append(f"{cs.accuracy[m]:.2f}" if m in cs.accuracy else "-")
        strong_cost = cs.cost_by_model.get(report.strong)
        cells.append(f"{strong_cost:.0f}" if strong_cost is not None else "-")
        if cs.spread is None:
            cells.append("-")
        elif cs.spread_reliable:
            cells.append(f"{cs.spread:+.2f}")
        else:
            cells.append(f"{cs.spread:+.2f}*")  # below the accuracy floor — see note
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
