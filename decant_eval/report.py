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

from .grading import JUDGE_ERROR_PREFIX
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
    # Tier the ranking and the cost column read from — `strong`, except on a run
    # that had no strong tier (weak-only), where it is the tier that did run.
    rank_model: str | None = None


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

    # The tier the ranking and the cost column are read from. Normally the
    # strong one. A run with no strong tier at all — a weak-only dev pass, which
    # is how the compression loop iterates cheaply, since weak-reader accuracy is
    # the binding constraint — would otherwise tie every conversion at "missing
    # strong" and order them arbitrarily, with no cost column at all.
    rank_model = strong if strong in models else (models[0] if len(models) == 1 else strong)

    # Rank: rank-tier accuracy (a conversion missing that tier sinks to the
    # bottom, not substituted by a mean), then cheapest on it, then — only when
    # the spread is reliable — tightest spread. None spread sorts last.
    def key(cs: ConversionScore):
        has_rank = rank_model in cs.accuracy
        acc = cs.accuracy.get(rank_model, 0.0)
        cost = cs.cost_by_model.get(rank_model, float("inf"))
        spread = cs.spread if (cs.spread is not None and cs.spread_reliable) else float("inf")
        return (0 if has_rank else 1, -acc, cost, spread)

    scores.sort(key=key)
    return Report(
        scores=scores, strong=strong, weak=weak, models=models,
        common_cases=sorted(scoring_cases), excluded_cases=excluded, comparable=comparable,
        rank_model=rank_model,
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


def regime_note(rows) -> str:
    """A '## Run regime' section: the reasoning configuration each model's rows
    were measured under, or "" when there is nothing to report.

    Accuracy is only comparable across rows measured the same way. Until Opus 5
    that was free -- every model the harness ran thought nothing when `thinking`
    was omitted, so `model` pinned the regime by itself. It no longer does, so
    the regime rides on the row (runner.Result) and this section states it.

    Three things get flagged, all of which otherwise average in silently:
      - more than one regime for a single model, which is what --resume across
        a config change produces (and _cells emits conversion-major, so the
        split falls on an arm boundary -- a per-arm difference, not noise);
      - rows predating the fields, which assert nothing about their own regime;
      - a model that thinks by default, where the arms think different amounts
        because thinking depth follows the input.
    """
    from .models import default_thinks

    if not rows:
        return ""
    combos: dict[str, set[tuple[int, str, str]]] = {}
    uninstrumented: dict[str, int] = {}
    for r in rows:
        m = r.model
        mt = getattr(r, "max_tokens", 0) or 0
        if mt:
            combos.setdefault(m, set()).add(
                (mt, getattr(r, "effort", "") or "", getattr(r, "thinking", "") or "")
            )
        else:
            uninstrumented[m] = uninstrumented.get(m, 0) + 1

    thinkers = sorted({r.model for r in rows if default_thinks(r.model) is True})
    unclassified = sorted({r.model for r in rows if default_thinks(r.model) is None})
    mixed = sorted(m for m, c in combos.items() if len(c) > 1)
    if not (combos or uninstrumented or thinkers or unclassified):
        return ""

    lines = ["## Run regime", "",
             "| model | max_tok | effort | thinking | default |",
             "| --- | --- | --- | --- | --- |"]
    for m in sorted(set(combos) | set(uninstrumented)):
        thinks = default_thinks(m)
        default = {True: "**thinks**", False: "no thinking", None: "**UNKNOWN**"}[thinks]
        for mt, effort, thinking in sorted(combos.get(m, set())):
            lines.append(
                f"| {m} | {mt} | {effort or '(not sent)'} | "
                f"{thinking or '(not sent)'} | {default} |"
            )
        if m in uninstrumented:
            lines.append(f"| {m} | ? | ? | ? | {default} |")
    lines.append("")
    lines.append(
        "_`(not sent)` means the parameter was omitted, so the regime is the "
        "model's own default -- see the `default` column. Both columns are "
        "needed to read a row._"
    )
    if mixed:
        lines.append(
            f"_WARNING: {', '.join(mixed)} ran under MORE THAN ONE regime in this "
            "file. Rows measured different ways are averaged together above, and "
            "`_cells` emits conversion-major, so a resumed run splits on an arm "
            "boundary -- expect the difference to look like an arm effect. Do not "
            "compare arms across this file._"
        )
    if uninstrumented:
        total = sum(uninstrumented.values())
        lines.append(
            f"_WARNING: {total} row(s) predate regime recording and assert nothing "
            "about how they were measured. They are only comparable to newer rows "
            "if the run config never changed -- which this file cannot confirm._"
        )
    if thinkers:
        lines.append(
            f"_WARNING: {', '.join(thinkers)} "
            f"{'run' if len(thinkers) > 1 else 'runs'} adaptive thinking when "
            "`thinking` is omitted. Thinking depth follows the input, so the arms think "
            "different amounts by construction; thinking also bills as output "
            "tokens with no separate usage field and is charged against "
            "max_tokens. Accuracy and the spread are confounded -- see models.py._"
        )
    if unclassified:
        lines.append(
            f"_WARNING: no thinking-default on file for {', '.join(unclassified)}. "
            "Whether these rows thought is unknown; classify the model in "
            "models.py before trusting the numbers._"
        )
    return "\n".join(lines)


def judge_failure_note(rows) -> str:
    """A warning listing rows whose score reflects the JUDGE failing rather than
    the answer being wrong -- an outage, or a verdict cut off at max_tokens.

    Both score 0, because there is no verdict to score, but neither is evidence
    about the representation. Worse, they are not evenly spread: the judge is
    reached for `open` questions and as the fallback for `exact` answers that
    missed a strict match, which is what a *weaker* representation produces. So
    judge failures land preferentially on the arms that need rescuing, and a
    silent one flatters the gap. Same reasoning as truncation_note, one layer
    further in."""
    hits = [r for r in rows if JUDGE_ERROR_PREFIX in str(getattr(r, "detail", ""))]
    if not hits:
        return ""
    lines = [
        "## Judge failures",
        "",
        f"_WARNING: {len(hits)} of {len(rows)} rows scored 0 because the judge did "
        "not return a usable verdict, not because the answer was wrong. The judge "
        "is the rescue path for answers that missed a strict match, so these "
        "concentrate on the weaker arms and widen the gap. Re-grade them "
        "(`regrade --judge MODEL`) before reading the scoreboard:_",
        "",
    ]
    for r in hits:
        lines.append(
            f"- `{r.case}` / `{r.question_id}` ({r.model}, {r.conversion}): {r.detail}"
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
    rank_model = report.rank_model or report.strong
    cost_label = "cost (strong tok)" if rank_model == report.strong else f"cost ({rank_model} tok)"
    header = ["conversion", *report.models, cost_label, "spread", "n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for cs in report.scores:
        cells = [cs.conversion]
        for m in report.models:
            # `!` — the mean includes calls that failed outright; see footnote.
            flag = "!" if m in cs.failed_calls else ""
            cells.append(f"{cs.accuracy[m]:.2f}{flag}" if m in cs.accuracy else "-")
        rank_cost = cs.cost_by_model.get(rank_model)
        cost_flag = "!" if rank_model in cs.failed_calls else ""
        cells.append(f"{rank_cost:.0f}{cost_flag}" if rank_cost is not None else "-")
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
