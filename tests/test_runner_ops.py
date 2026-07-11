"""Ops and metric-integrity tests for the fixes from the Fable review:
JSONL persistence + resume, the raw-PDF arena anchor, the memory-contamination
control arm, and the report's common-case / floor / missing-tier guards. Offline.
"""

import json
import tempfile
import unittest
from pathlib import Path

from decant_eval.corpus import Case, Question
from decant_eval.models import FakeModelClient
from decant_eval.report import build_report
from decant_eval.runner import (
    CONTROL, RAW, Result, load_completed, run_case, run_control, run_corpus,
)

STRONG, WEAK = "claude-opus-4-8", "claude-haiku-4-5"


def qs():
    return (
        Question(id="total", question="total?", gold="1250.00", type="numeric", tolerance=0.01),
        Question(id="vendor", question="vendor?", gold="Acme Corp", type="exact"),
    )


def make_case(tmp, name="c1", with_pdf=False):
    d = Path(tmp) / name
    (d / "conversions").mkdir(parents=True)
    (d / "questions.json").write_text(
        json.dumps({"questions": [
            {"id": "total", "question": "total?", "gold": "1250.00", "type": "numeric", "tolerance": 0.01},
            {"id": "vendor", "question": "vendor?", "gold": "Acme Corp", "type": "exact"},
        ]}), encoding="utf-8",
    )
    (d / "conversions" / "clean.md").write_text("Total: 1250.00 USD\nVendor: Acme Corp", encoding="utf-8")
    if with_pdf:
        (d / "source.pdf").write_bytes(b"%PDF-1.4 fake")
    return d


def answerer(model, system, prompt):
    if "total" in prompt.lower():
        return "1250.00"
    if "vendor" in prompt.lower():
        return "Acme Corp"
    return "NOT FOUND"


class TestJsonlPersistence(unittest.TestCase):
    def test_rows_stream_to_jsonl_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _load(make_case(tmp))
            path = Path(tmp) / "rows.jsonl"
            rows = run_case(case, client=FakeModelClient(answerer), models=[STRONG], jsonl_path=path)
            # one line per row, valid UTF-8 JSON
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(rows))
            reloaded, done = load_completed(path)
            self.assertEqual(len(reloaded), len(rows))
            self.assertIn((case.name, "clean", STRONG, "total"), done)

    def test_resume_skips_completed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _load(make_case(tmp))
            path = Path(tmp) / "rows.jsonl"
            client = FakeModelClient(answerer)
            run_case(case, client=client, models=[STRONG], jsonl_path=path)
            calls_after_first = len(client.calls)
            # Resume: every row is already done, so no new model calls happen.
            rows = run_corpus([case], client=client, models=[STRONG],
                              jsonl_path=path, resume=True)
            self.assertEqual(len(client.calls), calls_after_first)  # nothing re-run
            self.assertEqual(len(rows), 2)  # prior rows still returned


class TestRawArena(unittest.TestCase):
    def test_source_pdf_becomes_raw_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _load(make_case(tmp, with_pdf=True))
            client = FakeModelClient(answerer)
            rows = run_case(case, client=client, models=[STRONG])
            convs = {r.conversion for r in rows}
            self.assertIn(RAW, convs)
            self.assertIn("clean", convs)
            # the raw entry was fed as a PDF document, not extracted text
            self.assertTrue(any("[PDF source.pdf]" in prompt for _, prompt in client.calls))

    def test_no_raw_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _load(make_case(tmp, with_pdf=True))
            rows = run_case(case, client=FakeModelClient(answerer), models=[STRONG], raw_arena=False)
            self.assertNotIn(RAW, {r.conversion for r in rows})


class TestControlArm(unittest.TestCase):
    def test_memorized_answer_is_flagged(self):
        case = Case(name="c", questions=qs(), conversions={"clean": "x"})
        # A model that "remembers" the vendor with no document present.
        client = FakeModelClient(lambda m, s, p: "Acme Corp" if "vendor" in p.lower() else "NOT FOUND")
        rows = run_control(case, client=client, models=[STRONG])
        self.assertTrue(all(r.conversion == CONTROL for r in rows))
        by_q = {r.question_id: r for r in rows}
        self.assertTrue(by_q["vendor"].correct)   # answered from memory -> flagged
        self.assertFalse(by_q["total"].correct)


class TestReportIntegrity(unittest.TestCase):
    def _row(self, case, conv, model, score, tok=100):
        correct = score == 1.0
        return Result(case, conv, model, "q", "exact", correct, score, tok, 5, "a", "d")

    def test_restricts_to_common_cases(self):
        # conv "b" only exists in case1; case2 must be excluded from the comparison.
        rows = [
            self._row("case1", "a", STRONG, 1.0), self._row("case1", "b", STRONG, 1.0),
            self._row("case2", "a", STRONG, 0.0),  # would drag "a" down if counted
        ]
        rep = build_report(rows, strong=STRONG, weak=WEAK)
        self.assertTrue(rep.comparable)
        self.assertEqual(rep.excluded_cases, ["case2"])
        a = next(cs for cs in rep.scores if cs.conversion == "a")
        self.assertEqual(a.accuracy[STRONG], 1.0)  # scored only on the common case

    def test_missing_strong_tier_ranks_last(self):
        rows = [
            self._row("c", "weakonly", WEAK, 1.0),   # no strong-tier rows
            self._row("c", "full", STRONG, 0.5), self._row("c", "full", WEAK, 0.5),
        ]
        rep = build_report(rows, strong=STRONG, weak=WEAK)
        self.assertEqual(rep.scores[-1].conversion, "weakonly")

    def test_spread_below_floor_is_flagged(self):
        # both tiers fail equally: spread 0.0 but not a sign of robustness.
        rows = [self._row("c", "useless", STRONG, 0.0), self._row("c", "useless", WEAK, 0.0)]
        rep = build_report(rows, strong=STRONG, weak=WEAK)
        cs = rep.scores[0]
        self.assertEqual(cs.spread, 0.0)
        self.assertFalse(cs.spread_reliable)

    def test_markdown_is_console_safe_ascii(self):
        # finding 6: the printed report must survive a cp1252 Windows console.
        from decant_eval.report import to_markdown
        rows = [self._row("c1", "a", STRONG, 1.0), self._row("c1", "a", WEAK, 0.5),
                self._row("c2", "b", STRONG, 0.0)]  # forces excluded-case + floor notes
        md = to_markdown(build_report(rows, strong=STRONG, weak=WEAK))
        self.assertTrue(md.isascii(), "printed report must be pure ASCII for any console")


class TestSourceSlice(unittest.TestCase):
    """The `source` tag: question -> result row -> per-(case, source) report
    slice. This is the per-figure readout for the companion-PDF ablation."""

    def _row(self, conv, model, qid, score, source):
        return Result("c1", conv, model, qid, "exact", score == 1.0, score,
                      100, 5, "a", "d", source=source)

    def test_tag_flows_from_question_to_row(self):
        case = Case(
            name="c",
            questions=(
                Question(id="vendor", question="vendor?", gold="Acme Corp",
                         type="exact", source="figure-3"),
            ),
            conversions={"clean": "Vendor: Acme Corp"},
        )
        rows = run_case(case, client=FakeModelClient(answerer), models=[STRONG])
        self.assertEqual(rows[0].source, "figure-3")
        control = run_control(case, client=FakeModelClient(answerer), models=[STRONG])
        self.assertEqual(control[0].source, "figure-3")

    def test_pre_tagging_jsonl_still_loads(self):
        # A resume from an audit trail written before the field existed.
        old = {"case": "c", "conversion": "clean", "model": STRONG,
               "question_id": "q", "question_type": "exact", "correct": True,
               "score": 1.0, "input_tokens": 1, "output_tokens": 1,
               "answer": "a", "detail": "d"}  # no "source"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(old) + "\n", encoding="utf-8")
            rows, done = load_completed(path)
            self.assertEqual(rows[0].source, "")
            self.assertIn(("c", "clean", STRONG, "q"), done)

    def test_slice_groups_by_case_and_source(self):
        from decant_eval.report import build_source_scores, source_scores_markdown

        rows = [
            # figure-borne question: companion arm answers, plain arm doesn't
            self._row("decant", STRONG, "f1", 1.0, "figure-12"),
            self._row("decant-plain", STRONG, "f1", 0.0, "figure-12"),
            # text-borne question: both arms fine
            self._row("decant", STRONG, "t1", 1.0, "text"),
            self._row("decant-plain", STRONG, "t1", 1.0, "text"),
        ]
        scores = build_source_scores(rows)
        by = {(s.source, s.conversion): s for s in scores}
        self.assertEqual(by[("figure-12", "decant")].accuracy[STRONG], 1.0)
        self.assertEqual(by[("figure-12", "decant-plain")].accuracy[STRONG], 0.0)
        self.assertEqual(by[("text", "decant-plain")].accuracy[STRONG], 1.0)
        md = source_scores_markdown(scores, [STRONG])
        self.assertIn("figure-12", md)
        self.assertTrue(md.isascii())

    def test_untagged_rows_produce_no_section(self):
        from decant_eval.report import build_source_scores, source_scores_markdown

        rows = [self._row("decant", STRONG, "q", 1.0, "")]
        self.assertEqual(build_source_scores(rows), [])
        self.assertEqual(source_scores_markdown([], [STRONG]), "")


def _load(case_dir):
    from decant_eval.corpus import load_case
    return load_case(case_dir)


if __name__ == "__main__":
    unittest.main()
