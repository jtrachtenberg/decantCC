"""Grade a model's free-form answer against the gold answer.

Hybrid (the chosen methodology): grade programmatically where the answer is a
discrete value — a grader that can itself be wrong reintroduces the very
treacherous-degradation problem the eval exists to measure — and fall back to
an LLM judge only where wording genuinely varies.

    numeric  the answer's number vs. gold, within tolerance
    exact    normalized equality (or barely-longer containment); else the judge
    set      every gold item present as a whole word — score = fraction
    open     LLM judge → correct / partial / incorrect

The graders over-credited hedged, negated, and verbose answers — exactly what
models emit when reading a *corrupted* conversion — which would have rescued bad
conversions and inverted the signal the harness measures. The fixes:

  - Grade a *short* answer. The prompt demands a bare value; we take the final
    line (or an explicit "ANSWER: x"), so a paragraph of hedging can't smuggle
    the gold string past a containment check.
  - numeric takes the answer's *first* number, not any number in the text, so
    "800.00, not 1250.00" scores against 800, not 1250.
  - exact is equality, or containment only when the answer barely exceeds the
    gold; anything longer routes to the judge (when one is configured).
  - set matches on word boundaries, so gold "ship" is not found in "shipping".
  - the judge parser never keyword-guesses: an unparseable verdict is a logged
    incorrect, so "not correct" can't be read as "correct".

grade() returns (correct: bool, score: float in [0,1], detail: str). Detail is
ASCII-only — grading rows are written to logs and cp1252 Windows consoles.
"""

from __future__ import annotations

import json
import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s%.\-/]")
# First signed number, optional thousands separators and decimal.
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# An explicit "answer: x" / "final answer - x" line the model may have written.
_ANSWER_MARKER = re.compile(r"(?im)^\s*(?:final\s+)?answer\s*[:\-]\s*(.+?)\s*$")
# A bare verdict word, optionally "verdict: <word>" — the whole response, so a
# negated phrase like "not correct" is *not* accepted (it routes to unparseable).
_VERDICT_ONLY = re.compile(r'(?is)^\s*(?:verdict\s*[:=]\s*)?"?(correct|partial|incorrect)"?\.?\s*$')

# Containment (gold inside answer) counts only when the answer is at most this
# many normalized chars longer than the gold — "Acme Corp." yes, a sentence no.
_EXACT_SLACK = 8


def _norm(s) -> str:
    s = _PUNCT.sub(" ", str(s).lower())
    return _WS.sub(" ", s).strip()


def _numbers(s) -> list[float]:
    out = []
    for m in _NUM.findall(str(s)):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _short_answer(text: str) -> str:
    """The value the model actually answered with — an explicit ANSWER: line if
    present, else the last non-empty line. Keeps a hedged preamble out of the
    grade so containment/number checks see only the committed answer."""
    text = str(text).strip()
    marks = _ANSWER_MARKER.findall(text)
    if marks:
        return marks[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


def _boundary(term: str) -> re.Pattern:
    # Whole-token match: `ship` matches "ship" but not "shipping" or "township".
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")


def _grade_numeric(answer: str, gold, tolerance: float):
    try:
        target = float(str(gold).replace(",", ""))
    except ValueError:
        return False, 0.0, f"gold {gold!r} is not numeric"
    nums = _numbers(_short_answer(answer))
    if not nums:
        return False, 0.0, "no number in answer"
    n = nums[0]  # the committed value, not any number the model mentioned
    if abs(n - target) <= tolerance:
        return True, 1.0, f"answer {n} within +/-{tolerance} of {target}"
    return False, 0.0, f"answer {n} not within +/-{tolerance} of {target}"


def _grade_exact(answer: str, gold, *, question=None, judge=None, judge_model=""):
    g = _norm(gold)
    if not g:
        return False, 0.0, "gold string is empty"
    a = _norm(_short_answer(answer))
    if a == g:
        return True, 1.0, "exact match"
    if g in a and len(a) - len(g) <= _EXACT_SLACK:
        return True, 1.0, "answer contains gold (barely longer)"
    if judge is not None and question is not None:
        return _grade_open(answer, gold, question, judge, judge_model)
    return False, 0.0, "answer does not match gold"


def _grade_set(answer: str, gold):
    items = list(gold) if isinstance(gold, (list, tuple)) else [gold]
    a = _norm(answer)
    hits = [g for g in items if _norm(g) and _boundary(_norm(g)).search(a)]
    score = len(hits) / len(items) if items else 0.0
    return score == 1.0, score, f"{len(hits)}/{len(items)} items present"


_JUDGE_SYSTEM = (
    "You are a strict grader. Given a QUESTION, the GOLD answer, and a CANDIDATE "
    "answer, decide whether the candidate conveys the same factual content as the "
    "gold. Ignore wording, formatting, and extra detail; judge only factual "
    "agreement. Respond with a single JSON object: "
    '{"verdict": "correct" | "partial" | "incorrect", "reason": "<short>"}.'
)


def _grade_open(answer: str, gold, question: str, judge, judge_model: str):
    if judge is None:
        return False, 0.0, "open question skipped (no judge configured)"
    prompt = (
        f"QUESTION: {question}\n\nGOLD: {gold}\n\nCANDIDATE: {answer}\n\n"
        "Respond with the JSON object only."
    )
    try:
        res = judge.answer(model=judge_model, system=_JUDGE_SYSTEM, prompt=prompt, max_tokens=256)
    except Exception as exc:  # a judge outage must not crash a 400-question run
        return False, 0.0, f"judge error: {type(exc).__name__}: {exc}"
    verdict, reason = _parse_verdict(res.text)
    score = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(verdict, 0.0)
    return score == 1.0, score, f"judge: {verdict} - {reason}"


def _parse_verdict(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            v = str(obj.get("verdict", "")).lower().strip()
            if v in ("correct", "partial", "incorrect"):
                return v, str(obj.get("reason", ""))[:200]
        except json.JSONDecodeError:
            pass
    m = _VERDICT_ONLY.match(text)
    if m:
        return m.group(1).lower(), "bare verdict"
    # No guessing: substring "correct" lives inside "incorrect"/"not correct".
    return "incorrect", f"unparseable judge response: {text.strip()[:80]!r}"


def grade(question, answer: str, *, judge=None, judge_model: str = "claude-opus-4-8"):
    """(correct, score, detail) for one answer against one Question."""
    if question.type == "numeric":
        return _grade_numeric(answer, question.gold, question.tolerance)
    if question.type == "exact":
        return _grade_exact(
            answer, question.gold,
            question=question.question, judge=judge, judge_model=judge_model,
        )
    if question.type == "set":
        return _grade_set(answer, question.gold)
    if question.type == "open":
        return _grade_open(answer, question.gold, question.question, judge, judge_model)
    raise ValueError(f"unknown question type {question.type!r}")
