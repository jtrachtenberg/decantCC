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
from dataclasses import asdict, dataclass
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
    # "" when the model produced an answer that was graded; a machine-readable
    # failure class (today only CONTEXT_OVERFLOW) when the API call itself
    # failed and the row was scored 0 with no answer. Defaulted so rows from a
    # pre-status audit trail still load on --resume; load_completed re-derives
    # it for legacy failure rows.
    status: str = ""


def _key(case: str, conversion: str, model: str, question_id: str):
    return (case, conversion, model, question_id)


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
        done.add(_key(r.case, r.conversion, r.model, r.question_id))
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


def run_case(
    case: Case,
    *,
    client,
    models: list[str],
    judge=None,
    judge_model: str = "claude-opus-4-8",
    max_tokens: int = 512,
    raw_arena: bool = True,
    jsonl_path=None,
    done: set | None = None,
) -> list[Result]:
    done = done or set()
    sink = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None
    rows: list[Result] = []
    try:
        for conv_name, document in _arena_entries(case, raw_arena=raw_arena):
            for model in models:
                for q in case.questions:
                    if _key(case.name, conv_name, model, q.id) in done:
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
                    else:
                        status = ""
                        correct, score, detail = grade(
                            q, res.text, judge=judge, judge_model=judge_model
                        )
                        answer_text = res.text
                        in_tok, out_tok = res.input_tokens, res.output_tokens
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
            rows.append(Result(
                case=case.name, conversion=CONTROL, model=model,
                question_id=q.id, question_type=q.type, correct=correct, score=score,
                input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                answer=res.text, detail=detail, source=q.source,
            ))
    return rows
