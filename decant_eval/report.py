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
