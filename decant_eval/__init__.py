"""Decant eval harness — measure how well a document *conversion* transfers
meaning to an LLM reader, for the fewest tokens (the decantCC thesis).

The harness scores CACHED conversion outputs — it never runs the converters
itself — so it is engine-agnostic and needs no GPU: a converter (Decant,
MarkItDown, Docling, olmOCR, …) is just another `.md`/`.txt` file dropped into
a case's `conversions/` folder. See README.md for the design and the four
methodology decisions it rests on.
"""

from .corpus import Case, Question, load_corpus
from .grading import grade
from .models import AnswerResult, FakeModelClient, ModelClient
from .runner import Result, run_case, run_corpus
from .report import Report, build_report

__all__ = [
    "Case",
    "Question",
    "load_corpus",
    "grade",
    "AnswerResult",
    "FakeModelClient",
    "ModelClient",
    "Result",
    "run_case",
    "run_corpus",
    "Report",
    "build_report",
]
