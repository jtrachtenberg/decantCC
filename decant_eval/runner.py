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

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, fields, replace
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
# Other deterministic request failures -- retrying cannot succeed, so a
# --resume that re-raised them would crash-loop at the same row forever.
# A PDF over the per-request page cap (100 pages on 200K-context models):
PDF_TOO_MANY_PAGES = "pdf_too_many_pages"
# A request body over the API's size limit (HTTP 413):
REQUEST_TOO_LARGE = "request_too_large"
# The model answered but declined (stop_reason "refusal", a safety
# classifier). Its text is kept, but a refusal is not a wrong answer about the
# document, so the row is reported like a failed call rather than graded.
REFUSAL = "refusal"


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
    # The reasoning regime this row was measured under. `model` alone used to
    # pin it -- every model the harness ran thought nothing when `thinking` was
    # omitted -- but that stopped being true with Opus 5 (models.py), so the
    # regime is now recorded rather than inferred. "" means "not sent", which
    # resolves through models.default_thinks(model); max_tokens 0 means the row
    # predates this field. All three defaulted so pre-instrumentation trails
    # still load on --resume, and report.regime_note flags rows carrying the
    # defaults instead of letting them average in silently.
    #
    # These exist for --resume specifically: rows already written are carried
    # over untouched, so a file that spans a config change holds two regimes at
    # once. That is the same hazard regrade_rows exists to repair for the
    # grader, and _cells emits conversion-major, so an interrupted-and-resumed
    # run splits cleanly ON an arm boundary -- the config change lands as a
    # per-arm difference, which is the shape of a confound rather than noise.
    max_tokens: int = 0
    effort: str = ""
    thinking: str = ""
    # The model the API says served the call (see models.AnswerResult). "" on
    # rows written before it was recorded.
    served_model: str = ""
    # Content identity of what was measured, so --resume and regrade can tell
    # a row for today's conversion/question from one for a since-edited file.
    # sha256 hex of: the document the model saw (text, plus companion or
    # source PDF bytes; "" for the control arm), the question text as asked,
    # and the grading contract (type, gold, tolerance, flags). "" on rows
    # written before they were recorded -- those are reused unverified.
    doc_sha256: str = ""
    question_sha256: str = ""
    gold_sha256: str = ""
    # The judge consulted while grading this row, and what it cost. The judge
    # is billed like any other call; leaving it out understated spend.
    judge_model: str = ""
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


_RESULT_FIELDS = frozenset(f.name for f in fields(Result))


def _sha256(*parts: bytes) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


_FILE_SHA: dict[tuple[str, int, int], str] = {}


def _file_bytes_sha(path) -> str:
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _FILE_SHA:
        _FILE_SHA[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    return _FILE_SHA[key]


def document_sha256(document) -> str:
    """Content hash of an arena document (see _arena_entries); "" for None."""
    if document is None:
        return ""
    kind, payload = document
    if kind == "text":
        return _sha256(b"text", payload.encode("utf-8"))
    if kind == "text+pdf":
        text, path = payload
        return _sha256(b"text+pdf", text.encode("utf-8"), _file_bytes_sha(path).encode())
    return _sha256(b"pdf", _file_bytes_sha(payload).encode())


def question_sha256(q) -> str:
    """Hash of the question as the model is asked it."""
    return _sha256(q.question.encode("utf-8"))


def gold_sha256(q) -> str:
    """Hash of how an answer to `q` is graded."""
    contract = [q.type, q.gold, q.tolerance, bool(getattr(q, "sign_insensitive", False))]
    return _sha256(json.dumps(contract, sort_keys=True, default=str).encode("utf-8"))


def _key(case: str, conversion: str, model: str, question_id: str, repeat: int = 0):
    return (case, conversion, model, question_id, repeat)


def _truncated(res) -> bool:
    """The model ran out of output budget mid-answer. Distinct from a wrong
    answer: the representation may well have carried the fact and the model was
    simply cut off before it said so. The row is still graded (a truncated
    answer can already contain the value) but is flagged, so a run can never
    quietly report a budget artifact as a transfer failure."""
    return getattr(res, "stop_reason", "") == "max_tokens"


_PDF_PAGES = re.compile(r"\bpdf pages?\b|\bpages? .{0,40}\bpdf\b")


def _failure_status(exc: Exception) -> str | None:
    """The machine-readable Result.status when `exc` is an API failure the
    arena records as a scored row, else None — the error re-raises and crashes
    the run so --resume retries it (a transient outage must not be frozen into
    the audit trail as a permanent 0). Recorded failures are deterministic —
    retrying cannot succeed — and are about the representation's *size*:

      CONTEXT_OVERFLOW    the document exceeds the context window (e.g. a
                          98-page raw PDF at 201K tokens vs Haiku's 200K)
      REQUEST_TOO_LARGE   HTTP 413, the request body over the size limit
      PDF_TOO_MANY_PAGES  a PDF over the per-request page cap

    Classified by the SDK error's `status_code` where it has one, else by the
    API's message, so the runner stays SDK-free for offline tests. Any other
    400 (a malformed request is a harness bug) still raises."""
    msg = str(exc).lower()
    if "prompt is too long" in msg:
        return CONTEXT_OVERFLOW
    if getattr(exc, "status_code", None) == 413 or "request_too_large" in msg \
            or "request too large" in msg:
        return REQUEST_TOO_LARGE
    if _PDF_PAGES.search(msg) and re.search(r"maximum|exceed|at most|limit|too many", msg):
        return PDF_TOO_MANY_PAGES
    return None


class _MeteredJudge:
    """Wraps the judge client to count what grading one row cost."""

    def __init__(self, judge):
        self._judge = judge
        self.input_tokens = self.output_tokens = self.calls = 0

    def answer(self, **kwargs):
        res = self._judge.answer(**kwargs)
        self.calls += 1
        self.input_tokens += getattr(res, "input_tokens", 0) or 0
        self.output_tokens += getattr(res, "output_tokens", 0) or 0
        return res


def grade_metered(q, answer: str, *, judge=None, judge_model: str = ""):
    """grade(), plus (judge_model, judge_in, judge_out) for the row -- the
    model is "" when the judge was never called."""
    metered = _MeteredJudge(judge) if judge is not None else None
    correct, score, detail = grade(q, answer, judge=metered, judge_model=judge_model)
    if metered is None or not metered.calls:
        return correct, score, detail, "", 0, 0
    return correct, score, detail, judge_model, metered.input_tokens, metered.output_tokens


def _row_key(r: Result):
    return _key(r.case, r.conversion, r.model, r.question_id, r.repeat)


def load_completed(jsonl_path) -> tuple[list[Result], set]:
    """Rows already written to `jsonl_path`, and the set of their keys (for
    resume). Returns ([], empty set) when the file doesn't exist.

    One row per key: when a key appears more than once (a re-asked stale row,
    or two runs appended to one file) the LAST row wins and the count is
    reported on stderr -- averaging both would weight that cell twice.
    A malformed final line (a write cut off by a kill) is skipped with a
    warning so --resume can still start; a malformed line anywhere else is
    real corruption and raises. Keys this version doesn't know (a trail
    written by a newer harness) are ignored rather than fatal."""
    p = Path(jsonl_path)
    if not p.exists():
        return [], set()
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    by_key: dict[tuple, Result] = {}
    dupes = 0
    for i, line in enumerate(lines):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"warning: {p}: skipping a malformed final line (a write cut "
                      f"off mid-row?); that cell will be re-asked", file=sys.stderr)
                continue
            raise
        # Failure rows written before the status field carried only the detail
        # prefix; re-derive it so legacy audit trails (e.g. the 2026-07 shipped
        # runs) resume and re-report with failed calls still classified.
        if "status" not in d and str(d.get("detail", "")).startswith("context overflow"):
            d["status"] = CONTEXT_OVERFLOW
        r = Result(**{k: v for k, v in d.items() if k in _RESULT_FIELDS})
        k = _row_key(r)
        if k in by_key:
            dupes += 1
            del by_key[k]  # re-insert so file order follows the surviving row
        by_key[k] = r
    if dupes:
        print(f"warning: {p}: {dupes} duplicate row key(s); kept the last row "
              f"for each", file=sys.stderr)
    return list(by_key.values()), set(by_key)


def current_rows(rows, cases, *, models=None, raw_arena: bool = True, repeats=None):
    """The rows that measure today's corpus, and a count of the rest.

    A row counts only when its case, question (in the loaded split),
    conversion (or the control arm), and -- when given -- model and repeat
    index are all in the current configuration, and its recorded content
    hashes match today's files. Everything else stays in the JSONL but out of
    the scoreboard: a retired question, a deleted arm, an answer to a
    since-regenerated conversion.

    Returns (kept, counts) with counts = {"out_of_scope": n, "stale": n,
    "unverified": n}. `unverified` counts KEPT rows that predate content
    hashing, whose match with today's files cannot be checked."""
    docs: dict[tuple[str, str], str] = {}
    qs: dict[tuple[str, str], object] = {}
    for case in cases:
        for name, document in _arena_entries(case, raw_arena=raw_arena):
            docs[(case.name, name)] = document_sha256(document)
        docs[(case.name, CONTROL)] = ""
        for q in case.questions:
            qs[(case.name, q.id)] = q
    model_set = set(models) if models is not None else None
    kept: list[Result] = []
    counts = {"out_of_scope": 0, "stale": 0, "unverified": 0}
    for r in rows:
        q = qs.get((r.case, r.question_id))
        doc = docs.get((r.case, r.conversion))
        if (q is None or doc is None
                or (model_set is not None and r.model not in model_set)
                or (repeats is not None and r.repeat >= repeats)):
            counts["out_of_scope"] += 1
            continue
        if ((r.doc_sha256 and r.doc_sha256 != doc)
                or (r.question_sha256 and r.question_sha256 != question_sha256(q))
                or (r.gold_sha256 and r.gold_sha256 != gold_sha256(q))):
            counts["stale"] += 1
            continue
        if not (r.question_sha256 and r.gold_sha256
                and (r.doc_sha256 or r.conversion == CONTROL)):
            counts["unverified"] += 1
        kept.append(r)
    return kept, counts


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


def _answer_row(case_name, conv_name, document, model, q, rep, *, client, system,
                judge, judge_model, max_tokens, doc_sha) -> Result:
    """Ask one cell and build its row (shared by the arena and control arm)."""
    judge_used, judge_in, judge_out = "", 0, 0
    try:
        res = client.answer(
            model=model,
            system=system,
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
        # The call failed, so there is no response to read the regime
        # back from. The request was still built the same way, so the
        # row carries the client's defaults rather than a false blank.
        effort, thinking, served = "", "", ""
    else:
        answer_text = res.text
        in_tok, out_tok = res.input_tokens, res.output_tokens
        cache_read = res.cache_read_tokens
        cache_creation = res.cache_creation_tokens
        truncated = _truncated(res)
        effort = getattr(res, "effort", "")
        thinking = getattr(res, "thinking", "")
        served = getattr(res, "served_model", "") or ""
        if getattr(res, "stop_reason", "") == REFUSAL:
            # A safety decline says nothing about whether the document
            # carried the fact; grading it as a wrong answer would.
            status = REFUSAL
            correct, score = False, 0.0
            detail = "refusal: the model declined (stop_reason refusal) -- not graded"
        else:
            status = ""
            correct, score, detail, judge_used, judge_in, judge_out = grade_metered(
                q, res.text, judge=judge, judge_model=judge_model
            )
            if truncated:
                detail = f"truncated (max_tokens): {detail}"[:200]
    return Result(
        case=case_name,
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
        max_tokens=max_tokens,
        effort=effort,
        thinking=thinking,
        served_model=served,
        doc_sha256=doc_sha,
        question_sha256=question_sha256(q),
        gold_sha256=gold_sha256(q),
        judge_model=judge_used,
        judge_input_tokens=judge_in,
        judge_output_tokens=judge_out,
    )


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
    doc_shas: dict[str, str] = {}
    try:
        for conv_name, document, model, q, rep in _cells(
            case, models, repeats=repeats, raw_arena=raw_arena
        ):
            if _key(case.name, conv_name, model, q.id, rep) in done:
                continue
            if conv_name not in doc_shas:
                doc_shas[conv_name] = document_sha256(document)
            row = _answer_row(
                case.name, conv_name, document, model, q, rep,
                client=client, system=ANSWER_SYSTEM, judge=judge,
                judge_model=judge_model, max_tokens=max_tokens,
                doc_sha=doc_shas[conv_name],
            )
            rows.append(row)
            if sink is not None:
                sink.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
                sink.flush()
    finally:
        if sink is not None:
            sink.close()
    return rows


def run_corpus(cases: list[Case], *, jsonl_path=None, resume: bool = False,
               stats: dict | None = None, **kwargs) -> list[Result]:
    """Run every case; with `resume`, reuse the rows already in `jsonl_path`.

    Only rows that measure the current configuration are reused and returned
    (see current_rows): a retired question, a dropped arm or model, a repeat
    index beyond `repeats`, or a row whose conversion or question has since
    changed is left in the file but neither skipped nor reported -- a changed
    cell is asked again. `stats`, when given, receives current_rows' counts.

    Without `resume`, an existing non-empty `jsonl_path` is refused: appending
    a second run to it would put two rows on every key."""
    prior: list[Result] = []
    counts = {"out_of_scope": 0, "stale": 0, "unverified": 0}
    if jsonl_path and not resume:
        p = Path(jsonl_path)
        if p.exists() and p.stat().st_size > 0:
            raise FileExistsError(
                f"{p} already holds rows. Pass resume=True (--resume) to continue "
                f"it, or write to a new rows file; appending a fresh run would "
                f"double every cell it shares with the old one."
            )
    if jsonl_path and resume:
        loaded, _ = load_completed(jsonl_path)
        prior, counts = current_rows(
            loaded, cases,
            models=kwargs.get("models"),
            raw_arena=kwargs.get("raw_arena", True),
            repeats=max(1, kwargs.get("repeats", 1)),
        )
    if stats is not None:
        stats.update(counts)
    done = {_row_key(r) for r in prior}
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

    The gate and the refreshed row both follow the question as it is NOW: a
    question retyped from numeric to exact needs the judge like any other
    exact row, and the row's question_type, source tag (when the question has
    one) and gold hash are updated to match. Rows whose question is gone are returned unchanged
    (current_rows keeps them out of the report).

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
            and (q.type in OFFLINE_TYPES or judge is not None)
        )
        if not gradable:
            if q is not None and not row.status:
                skipped += 1
            out.append(row)
            continue
        correct, score, detail, jm, jin, jout = grade_metered(
            q, row.answer, judge=judge, judge_model=judge_model)
        if row.truncated:
            detail = f"truncated (max_tokens): {detail}"[:200]
        if score != row.score:
            changed += 1
        out.append(replace(
            row, correct=correct, score=score, detail=detail,
            # A tag added to the question since the run (retro-tagging)
            # reaches the row; an untagged question leaves the row's alone.
            question_type=q.type, source=q.source or row.source,
            gold_sha256=gold_sha256(q),
            # A fresh verdict replaces the old judge's; an offline one used none.
            judge_model=jm, judge_input_tokens=jin, judge_output_tokens=jout,
        ))
    return out, changed, skipped


def run_control(
    case: Case,
    *,
    client,
    models: list[str],
    judge=None,
    judge_model: str = "claude-opus-4-8",
    max_tokens: int = 512,
    jsonl_path=None,
    done: set | None = None,
) -> list[Result]:
    """The memory-contamination control arm: ask every question with NO document.
    Any question answered correctly here is answerable from the model's training
    data, so a conversion that "transfers" it proves nothing — flag it. Cheap
    (models × questions, no document tokens) and it doubles as the contamination
    audit for a public-document benchmark.

    Rows stream to `jsonl_path` like arena rows (conversion CONTROL) and keys
    in `done` are skipped, so --resume no longer re-bills the whole arm."""
    done = done or set()
    sink = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None
    rows: list[Result] = []
    try:
        for model in models:
            for q in case.questions:
                if _key(case.name, CONTROL, model, q.id, 0) in done:
                    continue
                row = _answer_row(
                    case.name, CONTROL, None, model, q, 0,
                    client=client, system=CONTROL_SYSTEM, judge=judge,
                    judge_model=judge_model, max_tokens=max_tokens, doc_sha="",
                )
                rows.append(row)
                if sink is not None:
                    sink.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
                    sink.flush()
    finally:
        if sink is not None:
            sink.close()
    return rows
