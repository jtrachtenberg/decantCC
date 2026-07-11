"""Load an eval corpus from disk.

A corpus is a directory of *cases*; each case is one source document plus the
question set authored against it and the candidate conversions to score:

    corpus/
      <case>/
        source.pdf            # the original (reference only; not fed to models)
        questions.json        # or questions.yaml (needs pyyaml)
        conversions/
          raw.md              # each file = one conversion in the arena
          decant.md
          decant.pdf          # optional figures companion: same stem as the
                              # conversion it belongs to; fed to the model
                              # alongside that conversion's text (see runner)
          markitdown.md
          docling.md

Hard rule (enforced by convention, not code): gold answers in questions.json
come from the SOURCE document, never from a conversion — otherwise the eval
measures conformance to a converter instead of correctness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Question types the grader understands (see grading.py).
QUESTION_TYPES = ("numeric", "exact", "set", "ordered_list", "open")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold: object  # str | list[str] | number, by type
    type: str = "exact"
    tolerance: float = 0.0  # numeric only: absolute tolerance on the value
    # Where in the source document the answer lives ("figure-12", "table-3",
    # "text") — free-form, case-scoped. Carried through to result rows so the
    # report can slice accuracy by answer location (e.g. do chart-borne
    # questions survive without the figures companion?). Empty = untagged.
    source: str = ""


@dataclass(frozen=True)
class Case:
    name: str
    questions: tuple[Question, ...]
    conversions: dict[str, str]  # conversion name -> markdown/text
    source: Path | None = None
    # conversion name -> its optional figures-companion PDF (same filename
    # stem), fed to the model alongside that conversion's text.
    companions: dict[str, Path] = field(default_factory=dict)


def _load_questions_file(path: Path) -> list[dict]:
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # optional dependency; JSON needs nothing
        except ModuleNotFoundError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                f"{path.name} is YAML but pyyaml isn't installed; "
                "use questions.json or `pip install pyyaml`"
            ) from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("questions", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of questions")
    return data


def _parse_questions(raw: list[dict], where: str) -> tuple[Question, ...]:
    out = []
    for i, q in enumerate(raw):
        qtype = q.get("type", "exact")
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"{where}: question {i} has unknown type {qtype!r}")
        if "id" not in q or "question" not in q or "gold" not in q:
            raise ValueError(f"{where}: question {i} needs id, question, and gold")
        out.append(
            Question(
                id=str(q["id"]),
                question=str(q["question"]),
                gold=q["gold"],
                type=qtype,
                tolerance=float(q.get("tolerance", 0.0)),
                source=str(q.get("source", "")),
            )
        )
    return tuple(out)


def _read_conversions(conv_dir: Path) -> dict[str, str]:
    """Non-empty .md/.txt files in conv_dir, keyed by filename stem. Empty or
    whitespace-only files are skipped so a placeholder or half-written arm
    (e.g. a decant.md the converter hasn't populated yet) cannot become a
    silently-blank arena entry that scores uniformly wrong in a billed run."""
    out: dict[str, str] = {}
    if conv_dir.is_dir():
        for p in sorted(conv_dir.iterdir()):
            if p.is_file() and p.suffix in (".md", ".txt"):
                text = p.read_text(encoding="utf-8")
                if text.strip():
                    out[p.stem] = text
    return out


def load_case(case_dir: Path) -> Case:
    case_dir = Path(case_dir)
    qfile = next(
        (case_dir / f"questions{ext}" for ext in (".json", ".yaml", ".yml") if (case_dir / f"questions{ext}").exists()),
        None,
    )
    if qfile is None:
        raise FileNotFoundError(f"{case_dir}: no questions.json/.yaml")
    questions = _parse_questions(_load_questions_file(qfile), qfile.name)

    conv_dir = case_dir / "conversions"
    if not conv_dir.is_dir():
        raise FileNotFoundError(f"{case_dir}: no conversions/ directory")
    conversions = _read_conversions(conv_dir)
    if not conversions:
        raise ValueError(f"{conv_dir}: no non-empty .md/.txt conversions found")

    # A PDF in conversions/ is a figures companion for the same-stem conversion
    # (e.g. decant.pdf rides with decant.md). An orphan PDF — no non-empty
    # matching conversion — is a loud error rather than a silent skip: a typo'd
    # stem would otherwise quietly drop the figures from a billed run.
    companions: dict[str, Path] = {}
    for p in sorted(conv_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".pdf":
            if p.stem in conversions:
                companions[p.stem] = p
            else:
                raise ValueError(
                    f"{p}: companion PDF has no non-empty {p.stem}.md/.txt "
                    "conversion to attach to"
                )

    source = next((p for p in case_dir.glob("source.*") if p.is_file()), None)
    return Case(
        name=case_dir.name, questions=questions, conversions=conversions,
        source=source, companions=companions,
    )


def load_corpus(corpus_dir: str | Path) -> list[Case]:
    """Every immediate subdirectory of corpus_dir that is a complete case:
    a questions file plus at least one conversion. Scaffold dirs that are
    still missing either half are skipped, so cases can be authored
    incrementally in any order (questions-first or conversions-first).
    load_case() stays strict — pointing it at an incomplete case is an error.
    """
    corpus_dir = Path(corpus_dir)
    cases = []
    for child in sorted(corpus_dir.iterdir()):
        if not child.is_dir():
            continue
        has_questions = any(
            (child / f"questions{ext}").exists() for ext in (".json", ".yaml", ".yml")
        )
        has_conversions = bool(_read_conversions(child / "conversions"))
        if has_questions and has_conversions:
            cases.append(load_case(child))
    if not cases:
        raise ValueError(f"{corpus_dir}: no cases found")
    return cases
