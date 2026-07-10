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
          markitdown.md
          docling.md

Hard rule (enforced by convention, not code): gold answers in questions.json
come from the SOURCE document, never from a conversion — otherwise the eval
measures conformance to a converter instead of correctness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Question types the grader understands (see grading.py).
QUESTION_TYPES = ("numeric", "exact", "set", "open")


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold: object  # str | list[str] | number, by type
    type: str = "exact"
    tolerance: float = 0.0  # numeric only: absolute tolerance on the value


@dataclass(frozen=True)
class Case:
    name: str
    questions: tuple[Question, ...]
    conversions: dict[str, str]  # conversion name -> markdown/text
    source: Path | None = None


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
            )
        )
    return tuple(out)


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
    conversions = {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted(conv_dir.iterdir())
        if p.is_file() and p.suffix in (".md", ".txt")
    }
    if not conversions:
        raise ValueError(f"{conv_dir}: no .md/.txt conversions found")

    source = next((p for p in case_dir.glob("source.*") if p.is_file()), None)
    return Case(name=case_dir.name, questions=questions, conversions=conversions, source=source)


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
        conv_dir = child / "conversions"
        has_conversions = conv_dir.is_dir() and any(
            p.is_file() and p.suffix in (".md", ".txt") for p in conv_dir.iterdir()
        )
        if has_questions and has_conversions:
            cases.append(load_case(child))
    if not cases:
        raise ValueError(f"{corpus_dir}: no cases found")
    return cases
