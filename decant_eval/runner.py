"""Run the arena: for each case × conversion × target model × question, feed
the conversion + question to the model, grade the answer, record a row.

Per-answer token cost comes from the response's own `usage` (exact and free),
so no separate count_tokens call is needed during a run.

Rows are appended to a JSONL file (UTF-8) *as they complete*, not held in memory
until the end — a judge outage or crash at question 380/400 must not discard the
whole run, and the graded-testing pass needs the per-answer audit trail. A run
can resume from that JSONL: rows already present are skipped, not re-billed.

The source PDF, when present, enters the arena as the implicit **raw upload**
entry — the baseline ("is this conversion better than doing nothing?") the whole
thesis is measured against. It's fed as a document block, not extracted text.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .corpus import Case
from .grading import grade

# The model is told to answer strictly from the provided document — the eval
# measures what the *representation* carries, not the model's prior knowledge.
ANSWER_SYSTEM = (
    "Answer the question using ONLY the information in the DOCUMENT provided. "
    "Do not use outside knowledge. If the document does not contain the answer, "
    "reply exactly: NOT FOUND. Answer concisely — the value or fact asked for, "
    "nothing more."
)

# Control arm: no document at all. Any question answered correctly here was
# answered from the model's memory, not the representation — see run_control.
CONTROL_SYSTEM = (
    "Answer the question from your own knowledge. If you do not know the answer, "
    "reply exactly: NOT FOUND. Answer concisely — the value or fact asked for, "
    "nothing more."
)

RAW = "raw"  # arena name for the source-PDF baseline
CONTROL = "(memory)"  # arena name for the no-document control arm

# Result.status for a row whose model call failed because the document exceeds
# the target model's context window. Scored 0 by design (see _failure_status),
# but the report shows it as a failed call, not a graded wrong answer.
CONTEXT_OVERFLOW = "context_overflow"


def _question_prompt(question: str) -> str:
    return f"QUESTION: {question}\n\nANSWER:"


@dataclass(frozen=True)
class Result:
    case: str
    conversion: str
    model: str
    question_id: str
    question_type: str
    correct: bool
    score: float
    input_tokens: int
    output_tokens: int
    answer: str
    detail: str
    # The question's answer-location tag (Question.source). Defaulted so rows
    # from a pre-tagging JSONL audit trail still load on --resume.
    source: str = ""
    # True when the model hit the output budget mid-answer (stop_reason
    # "max_tokens") rather than finishing. Such a row's score is about the
    # budget, not the representation — see _truncated. Defaulted for --resume.
    truncated: bool = False
    # 0-based sample index when a question is asked more than once (--repeats).
    # Part of the row key, so resume never conflates two samples of one cell.
    # Defaulted so rows from a pre-repeats JSONL still load.
    repeat: int = 0
    # The cached portions of input_tokens. Needed to price a run: reads bill at
    # ~0.1x and writes at ~1.25x, so input_tokens alone overstates spend several
    # fold on a cached run. 0 on rows written before these were recorded, which
    # makes those rows price as if fully uncached — an upper bound, flagged as
    # such by report.cost_summary rather than reported as fact.
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # "" when the model produced an answer that was graded; a machine-readable
    # failure class (today only CONTEXT_OVERFLOW) when the API call itself
    # failed and the row was scored 0 with no answer. Defaulted so rows from a
    # pre-status audit trail still load on --resume; load_completed re-derives
    # it for legacy failure rows.
    status: str = ""


def _key(case: str, conversion: str, model: str, question_id: str, repeat: int = 0):
    return (case, conversion, model, question_id, repeat)


def _truncated(res) -> bool:
    """The model ran out of output budget mid-answer. Distinct from a wrong
    answer: the representation may well have carried the fact and the model was
    simply cut off before it said so. The row is still graded (a truncated
    answer can already contain the value) but is flagged, so a run can never
    quietly report a budget artifact as a transfer failure."""
    return getattr(res, "stop_reason", "") == "max_tokens"


def _failure_status(exc: Exception) -> str | None:
    """The machine-readable Result.status when `exc` is an API failure the
    arena records as a scored row, else None — the error re-raises and crashes
    the run so --resume retries it (a transient outage must not be frozen into
    the audit trail as a permanent 0). CONTEXT_OVERFLOW is recorded: a
    representation that doesn't fit the target model's context window is a
    transfer failure, not an ops error, and deterministic — retrying cannot
    succeed (e.g. a 98-page raw PDF at 201K tokens vs Haiku's 200K). Matched on
    the API's message so the runner stays SDK-free for offline tests."""
    if "prompt is too long" in str(exc).lower():
        return CONTEXT_OVERFLOW
    return None


def load_completed(jsonl_path) -> tuple[list[Result], set]:
    """Rows already written to `jsonl_path`, and the set of their keys (for
    resume). Returns ([], empty set) when the file doesn't exist."""
    rows: list[Result] = []
    done: set = set()
    p = Path(jsonl_path)
    if not p.exists():
        return rows, done
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        # Failure rows written before the status field carried only the detail
        # prefix; re-derive it so legacy audit trails (e.g. the 2026-07 shipped
        # runs) resume and re-report with failed calls still classified.
        if "status" not in d and str(d.get("detail", "")).startswith("context overflow"):
            d["status"] = CONTEXT_OVERFLOW
        r = Result(**d)
        rows.append(r)
        done.add(_key(r.case, r.conversion, r.model, r.question_id, r.repeat))
    return rows, done


def _arena_entries(case: Case, *, raw_arena: bool):
    """(name, document) per conversion in the arena. document is ("text", md)
    for a conversion, ("text+pdf", (md, figures_path)) for a conversion with a
    same-stem figures-companion PDF (e.g. decant.md + decant.pdf), or
    ("pdf", path) for the source-PDF raw baseline."""
    entries = []
    for name, text in case.conversions.items():
        companion = case.companions.get(name)
        if companion is not None:
            entries.append((name, ("text+pdf", (text, companion))))
        else:
            entries.append((name, ("text", text)))
    if raw_arena and case.source is not None and case.source.suffix.lower() == ".pdf":
        if RAW not in case.conversions:  # don't shadow an explicit raw.md
            entries.append((RAW, ("pdf", case.source)))
    return entries


def _cells(case: Case, models: list[str], *, repeats: int, raw_arena: bool):
    """Every (conversion, document, model, question, repeat) the run must ask.

    `repeats` > 1 asks each question that many times. Sampling cannot be pinned
    on the strong tier (temperature is rejected on Opus 4.7+ — see models.py),
    so a single sample per cell is one draw from a distribution, not a
    measurement: on a 10-question case one flipped answer moves accuracy 0.10,
    which is the same size as the effects the arena is trying to resolve.
    Repeats are the only variance control available; the report averages them."""
    for conv_name, document in _arena_entries(case, raw_arena=raw_arena):
        for model in models:
            for q in case.questions:
                for rep in range(max(1, repeats)):
                    yield conv_name, document, model, q, rep


def run_case(
    case: Case,
    *,
    client,
    models: list[str],
    judge=None,
    judge_model: str = "claude-opus-4-8",
    max_tokens: int = 512,
    repeats: int = 1,
    raw_arena: bool = True,
    jsonl_path=None,
    done: set | None = None,
) -> list[Result]:
    done = done or set()
    sink = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None
    rows: list[Result] = []
    try:
        for conv_name, document, model, q, rep in _cells(
            case, models, repeats=repeats, raw_arena=raw_arena
        ):
            if _key(case.name, conv_name, model, q.id, rep) in done:
                continue
            try:
                res = client.answer(
                    model=model,
                    system=ANSWER_SYSTEM,
                    prompt=_question_prompt(q.question),
                    max_tokens=max_tokens,
                    document=document,
                )
            except Exception as exc:
                status = _failure_status(exc)
                if status is None:
                    raise
                answer_text = ""
                correct, score = False, 0.0
                detail = f"{status.replace('_', ' ')}: {exc}"[:200]
                in_tok, out_tok = 0, 0
                cache_read = cache_creation = 0
                truncated = False
            else:
                status = ""
                correct, score, detail = grade(
                    q, res.text, judge=judge, judge_model=judge_model
                )
                answer_text = res.text
                in_tok, out_tok = res.input_tokens, res.output_tokens
                cache_read = res.cache_read_tokens
                cache_creation = res.cache_creation_tokens
                truncated = _truncated(res)
                if truncated:
                    detail = f"truncated (max_tokens): {detail}"[:200]
            row = Result(
                case=case.name,
                conversion=conv_name,
                model=model,
                question_id=q.id,
                question_type=q.type,
                correct=correct,
                score=score,
                input_tokens=in_tok,
                output_tokens=out_tok,
                answer=answer_text,
                detail=detail,
                source=q.source,
                truncated=truncated,
                repeat=rep,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_creation,
                status=status,
            )
            rows.append(row)
            if sink is not None:
                sink.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
                sink.flush()
    finally:
        if sink is not None:
            sink.close()
    return rows


def run_corpus(cases: list[Case], *, jsonl_path=None, resume: bool = False, **kwargs) -> list[Result]:
    prior: list[Result] = []
    done: set = set()
    if jsonl_path and resume:
        prior, done = load_completed(jsonl_path)
    rows = list(prior)
    for case in cases:
        rows.extend(
            run_case(case, jsonl_path=jsonl_path, done=done, **kwargs)
        )
    return rows


# Question types graded purely programmatically — `grade` never reaches for the
# judge on these, so they re-grade offline with no API call and no risk of a
# different verdict than the run would have produced. `exact` falls back to the
# judge when the answer isn't a match, and `open` is judge-only: re-grading
# either without a judge would turn judge-awarded passes into spurious zeros,
# so regrade_rows leaves them alone unless a judge is supplied.
OFFLINE_TYPES = ("numeric", "set", "ordered_list")


def regrade_rows(rows, cases, *, judge=None, judge_model: str = ""):
    """Re-grade `rows` against `cases`' current questions.

    The audit trail exists so a grader fix can be applied to answers already
    paid for. It is also the only way to repair a resumed run: rows carried
    over by --resume keep the verdict the *old* grader gave them, so a file
    that spans a grader change holds two regimes at once and averages across
    them silently.

    Returns (new_rows, changed, skipped) — `skipped` counts rows left at their
    stored verdict because grading them would have needed a judge that wasn't
    supplied. Rows whose call failed (`status` set) keep their verdict too and
    are not counted as skipped: there is no answer text to grade, and the
    failure is the finding. load_completed re-derives `status` for legacy rows,
    so this classifies pre-status audit trails correctly too."""
    by_case: dict[str, dict[str, object]] = {}
    for case in cases:
        by_case[case.name] = {q.id: q for q in case.questions}

    out: list[Result] = []
    changed = skipped = 0
    for row in rows:
        q = by_case.get(row.case, {}).get(row.question_id)
        gradable = (
            q is not None
            and not row.status
            and (row.question_type in OFFLINE_TYPES or judge is not None)
        )
        if not gradable:
            if q is not None and not row.status:
                skipped += 1
            out.append(row)
            continue
        correct, score, detail = grade(q, row.answer, judge=judge, judge_model=judge_model)
        if score != row.score:
            changed += 1
        out.append(replace(row, correct=correct, score=score, detail=detail))
    return out, changed, skipped


def run_control(
    case: Case,
    *,
    client,
    models: list[str],
    judge=None,
    judge_model: str = "claude-opus-4-8",
    max_tokens: int = 512,
) -> list[Result]:
    """The memory-contamination control arm: ask every question with NO document.
    Any question answered correctly here is answerable from the model's training
    data, so a conversion that "transfers" it proves nothing — flag it. Cheap
    (models × questions, no document tokens) and it doubles as the contamination
    audit for a public-document benchmark."""
    rows: list[Result] = []
    for model in models:
        for q in case.questions:
            res = client.answer(
                model=model, system=CONTROL_SYSTEM,
                prompt=_question_prompt(q.question), max_tokens=max_tokens, document=None,
            )
            correct, score, detail = grade(q, res.text, judge=judge, judge_model=judge_model)
            truncated = _truncated(res)
            if truncated:
                detail = f"truncated (max_tokens): {detail}"[:200]
            rows.append(Result(
                case=case.name, conversion=CONTROL, model=model,
                question_id=q.id, question_type=q.type, correct=correct, score=score,
                input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                answer=res.text, detail=detail, source=q.source, truncated=truncated,
                cache_read_tokens=res.cache_read_tokens,
                cache_creation_tokens=res.cache_creation_tokens,
            ))
    return rows
