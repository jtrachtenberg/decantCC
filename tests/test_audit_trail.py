"""Audit-trail integrity: what --resume and regrade may carry into a report.

The JSONL trail outlives the corpus it was measured against -- questions get
retired, arms deleted, conversions regenerated -- so reusing a row is only
sound when it still measures today's files. These tests cover the 2026-09
review findings on that path (B2, B7, B8, B10, B13, B16, B23). Offline.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from decant_eval.cli import main
from decant_eval.corpus import Case, Question, load_corpus
from decant_eval.models import FakeModelClient
from decant_eval.runner import (
    CONTROL, PDF_TOO_MANY_PAGES, REFUSAL, REQUEST_TOO_LARGE, Result, current_rows,
    load_completed, regrade_rows, run_case, run_control, run_corpus,
)

STRONG, WEAK = "claude-opus-4-8", "claude-haiku-4-5"


def write_case(root, name="c1", questions=None, conversions=None):
    d = Path(root) / name
    (d / "conversions").mkdir(parents=True, exist_ok=True)
    questions = questions or [
        {"id": "total", "question": "total?", "gold": "1250.00", "type": "numeric",
         "tolerance": 0.01},
        {"id": "vendor", "question": "vendor?", "gold": "Acme Corp", "type": "exact"},
    ]
    (d / "questions.json").write_text(json.dumps({"questions": questions}), encoding="utf-8")
    for stem, text in (conversions or {"clean": "Total: 1250.00\nVendor: Acme Corp"}).items():
        (d / "conversions" / f"{stem}.md").write_text(text, encoding="utf-8")
    return d


def answerer(model, system, prompt):
    p = prompt.lower()
    if "total" in p:
        return "1250.00"
    if "vendor" in p:
        return "Acme Corp"
    return "NOT FOUND"


class TestResumeCarriesOnlyCurrentRows(unittest.TestCase):
    """B2: --resume fed every row in the file to the report."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "corpus"
        self.rows = Path(self.tmp.name) / "rows.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, client, **kw):
        return run_corpus(load_corpus(self.root), client=client, models=[STRONG],
                          jsonl_path=self.rows, **kw)

    def test_rows_carry_content_hashes(self):
        write_case(self.root)
        rows = self._run(FakeModelClient(answerer))
        for r in rows:
            self.assertEqual(len(r.doc_sha256), 64)
            self.assertEqual(len(r.question_sha256), 64)
            self.assertEqual(len(r.gold_sha256), 64)

    def test_retired_question_is_not_reported_on_resume(self):
        write_case(self.root)
        self._run(FakeModelClient(answerer))
        write_case(self.root, questions=[
            {"id": "total", "question": "total?", "gold": "1250.00", "type": "numeric",
             "tolerance": 0.01},
            {"id": "vendor", "question": "vendor?", "gold": "Acme Corp", "type": "exact",
             "split": "retired"},
        ])
        stats = {}
        client = FakeModelClient(answerer)
        rows = self._run(client, resume=True, stats=stats)
        self.assertEqual([r.question_id for r in rows], ["total"])
        self.assertEqual(stats["out_of_scope"], 1)
        self.assertEqual(client.calls, [])

    def test_regenerated_conversion_is_asked_again(self):
        write_case(self.root)
        self._run(FakeModelClient(answerer))
        write_case(self.root, conversions={"clean": "Total: 1250.00 (regenerated)\nVendor: Acme Corp"})
        stats = {}
        client = FakeModelClient(answerer)
        rows = self._run(client, resume=True, stats=stats)
        self.assertEqual(len(client.calls), 2)       # both cells re-asked
        self.assertEqual(stats["stale"], 2)
        self.assertEqual(len(rows), 2)               # the new rows only
        reloaded, _ = load_completed(self.rows)
        self.assertEqual(len(reloaded), 2)           # last row per key wins

    def test_changed_gold_is_asked_again(self):
        write_case(self.root)
        self._run(FakeModelClient(answerer))
        write_case(self.root, questions=[
            {"id": "total", "question": "total?", "gold": "1300.00", "type": "numeric"},
            {"id": "vendor", "question": "vendor?", "gold": "Acme Corp", "type": "exact"},
        ])
        client = FakeModelClient(answerer)
        self._run(client, resume=True)
        self.assertEqual(len(client.calls), 1)

    def test_legacy_rows_without_hashes_are_reused_and_counted(self):
        write_case(self.root)
        self._run(FakeModelClient(answerer))
        legacy = [json.loads(ln) for ln in self.rows.read_text().splitlines()]
        for d in legacy:
            for k in ("doc_sha256", "question_sha256", "gold_sha256"):
                d.pop(k)
        self.rows.write_text("".join(json.dumps(d) + "\n" for d in legacy))
        stats = {}
        client = FakeModelClient(answerer)
        rows = self._run(client, resume=True, stats=stats)
        self.assertEqual(client.calls, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(stats["unverified"], 2)

    def test_dropped_model_and_extra_repeats_are_out_of_scope(self):
        write_case(self.root)
        run_corpus(load_corpus(self.root), client=FakeModelClient(answerer),
                   models=[STRONG, WEAK], repeats=2, jsonl_path=self.rows)
        stats = {}
        rows = self._run(FakeModelClient(answerer), resume=True, stats=stats)
        self.assertEqual({(r.model, r.repeat) for r in rows}, {(STRONG, 0)})
        self.assertEqual(stats["out_of_scope"], 6)


class TestFreshRunRefusesExistingTrail(unittest.TestCase):
    """B8: a run without --resume appended to an existing rows file."""

    def test_library_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_case(Path(tmp) / "corpus")
            cases = load_corpus(Path(tmp) / "corpus")
            path = Path(tmp) / "rows.jsonl"
            run_corpus(cases, client=FakeModelClient(answerer), models=[STRONG], jsonl_path=path)
            with self.assertRaises(FileExistsError):
                run_corpus(cases, client=FakeModelClient(answerer), models=[STRONG],
                           jsonl_path=path)

    def test_duplicate_keys_collapse_to_the_last_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Case(name="c", conversions={"clean": "x"}, questions=(
                Question(id="q", question="?", gold="1", type="numeric"),))
            path = Path(tmp) / "rows.jsonl"
            run_case(case, client=FakeModelClient(lambda m, s, p: "2"), models=[STRONG],
                     jsonl_path=path)
            run_case(case, client=FakeModelClient(lambda m, s, p: "1"), models=[STRONG],
                     jsonl_path=path)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                rows, _ = load_completed(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].score, 1.0)
            self.assertIn("duplicate", err.getvalue())


class TestJsonlRobustness(unittest.TestCase):
    """B23."""

    def _write(self, tmp, text):
        path = Path(tmp) / "rows.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def _good(self):
        return json.dumps(asdict(Result(
            case="c", conversion="clean", model=STRONG, question_id="q",
            question_type="numeric", correct=True, score=1.0, input_tokens=1,
            output_tokens=1, answer="1", detail="")))

    def test_truncated_last_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, self._good() + "\n" + '{"case": "c", "conv')
            with contextlib.redirect_stderr(io.StringIO()) as err:
                rows, _ = load_completed(path)
            self.assertEqual(len(rows), 1)
            self.assertIn("malformed final line", err.getvalue())

    def test_malformed_middle_line_still_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, self._good() + "\n{oops\n" + self._good() + "\n")
            with self.assertRaises(json.JSONDecodeError):
                load_completed(path)

    def test_unknown_keys_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = json.loads(self._good())
            d["field_from_the_future"] = 1
            rows, _ = load_completed(self._write(tmp, json.dumps(d) + "\n"))
            self.assertEqual(len(rows), 1)


class TestRegradeFreshness(unittest.TestCase):
    """B10: regrade gated on the stored type and kept stale type/source."""

    def test_retyped_question_is_gated_on_its_new_type(self):
        case = Case(name="c", conversions={"clean": "x"}, questions=(
            Question(id="q", question="?", gold="Acme Corp", type="exact", source="table-2"),))
        row = Result(case="c", conversion="clean", model=STRONG, question_id="q",
                     question_type="numeric", correct=True, score=1.0, input_tokens=1,
                     output_tokens=1, answer="It was Acme Corp, per the header.", detail="")
        new, changed, skipped = regrade_rows([row], [case])
        # Now exact: needs a judge that isn't here, so it is skipped, not zeroed.
        self.assertEqual((changed, skipped), (0, 1))
        self.assertEqual(new[0].score, 1.0)

    def test_regraded_row_takes_the_current_type_and_tag(self):
        case = Case(name="c", conversions={"clean": "x"}, questions=(
            Question(id="q", question="?", gold="1.6", type="numeric", source="table-10"),))
        row = Result(case="c", conversion="clean", model=STRONG, question_id="q",
                     question_type="exact", correct=False, score=0.0, input_tokens=1,
                     output_tokens=1, answer="1.60", detail="")
        new, _, _ = regrade_rows([row], [case])
        self.assertEqual(new[0].question_type, "numeric")
        self.assertEqual(new[0].source, "table-10")
        self.assertEqual(new[0].score, 1.0)


class TestRegradeCli(unittest.TestCase):
    """B7 and B2 via the `regrade` command."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.w = Path(self.tmp.name)
        write_case(self.w / "corp", questions=[
            {"id": "live", "question": "live?", "gold": "1.6", "type": "numeric"},
            {"id": "gone", "question": "gone?", "gold": "5", "type": "numeric",
             "split": "retired"},
        ], conversions={"a": "x"})
        base = {"case": "c1", "conversion": "a", "model": STRONG,
                "question_type": "numeric", "correct": False, "score": 0.0,
                "input_tokens": 1, "output_tokens": 1, "detail": "", "max_tokens": 512}
        self.inp = self.w / "report.md.jsonl"
        self.inp.write_text("".join(
            json.dumps({**base, "question_id": qid, "answer": ans}) + "\n"
            for qid, ans in [("live", "22.19 - 20.59 = 1.60"), ("gone", "5")]))

    def tearDown(self):
        self.tmp.cleanup()

    def _main(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
            code = main(argv)
        return code, out.getvalue()

    def test_refuses_to_overwrite_its_input(self):
        before = self.inp.read_text()
        with self.assertRaises(SystemExit) as cm:
            self._main(["regrade", "--corpus", str(self.w / "corp"),
                        "--rows", str(self.inp), "--out", str(self.w / "report.md")])
        self.assertEqual(cm.exception.code, 2)
        self.assertEqual(self.inp.read_text(), before)

    def test_retired_question_is_kept_in_rows_but_not_reported(self):
        code, _ = self._main(["regrade", "--corpus", str(self.w / "corp"),
                              "--rows", str(self.inp), "--out", str(self.w / "new.md")])
        self.assertEqual(code, 0)
        out_rows, _ = load_completed(self.w / "new.md.jsonl")
        self.assertEqual(sorted(r.question_id for r in out_rows), ["gone", "live"])
        md = (self.w / "new.md").read_text()
        self.assertIn("1 row(s) in the rows file are outside", md)
        self.assertIn("| a | 1.00 |", md)

    def test_exclude_arm_drops_rows_from_the_report(self):
        code, _ = self._main(["regrade", "--corpus", str(self.w / "corp"),
                              "--rows", str(self.inp), "--out", str(self.w / "new.md"),
                              "--exclude-arm", "a"])
        self.assertEqual(code, 0)
        md = (self.w / "new.md").read_text()
        self.assertNotIn("| a |", md)
        self.assertIn("2 row(s) in the rows file are outside", md)

    def test_split_with_no_cases_is_a_message_not_a_traceback(self):
        code, out = self._main(["regrade", "--corpus", str(self.w / "corp"), "--split", "dev",
                                "--rows", str(self.inp), "--out", str(self.w / "new.md")])
        self.assertEqual(code, 1)
        self.assertIn("nothing to re-grade", out)


class TestControlArmPersistence(unittest.TestCase):
    """B13: the control arm was never written to the trail, re-billed on every
    resume, and left out of the cost table; any credit is now flagged."""

    def test_control_rows_stream_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Case(name="c", conversions={"clean": "x"}, questions=(
                Question(id="q", question="?", gold="1", type="numeric"),))
            path = Path(tmp) / "rows.jsonl"
            client = FakeModelClient(lambda m, s, p: "NOT FOUND")
            run_control(case, client=client, models=[STRONG], jsonl_path=path)
            rows, done = load_completed(path)
            self.assertEqual([r.conversion for r in rows], [CONTROL])
            run_control(case, client=client, models=[STRONG], jsonl_path=path, done=done)
            self.assertEqual(len(client.calls), 1)

    def test_cli_run_persists_control_and_reports_partial_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corp"
            write_case(root, questions=[
                {"id": "frameworks", "question": "frameworks?", "gold": ["GRI", "SASB", "TCFD"],
                 "type": "set"},
            ], conversions={"clean": "GRI SASB TCFD"})
            fake = FakeModelClient(lambda m, s, p: "GRI, SASB and TCFD" if "DOCUMENT" in p
                                   else "GRI and SASB")
            out = Path(tmp) / "r.md"
            with mock.patch("decant_eval.cli.AnthropicModelClient", return_value=fake), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["run", "--corpus", str(root), "--out", str(out)])
                calls = len(fake.calls)
                main(["run", "--corpus", str(root), "--out", str(out), "--resume"])
            self.assertEqual(len(fake.calls), calls)  # nothing re-billed, control included
            rows, _ = load_completed(Path(f"{out}.jsonl"))
            self.assertIn(CONTROL, {r.conversion for r in rows})
            md = out.read_text()
            self.assertIn("earned credit with NO document", md)
            self.assertIn("score 0.67", md)

    def test_cli_refuses_to_append_a_fresh_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "corp"
            write_case(root)
            out = Path(tmp) / "r.md"
            Path(f"{out}.jsonl").write_text("{}\n")
            with open(os.devnull, "w") as null, contextlib.redirect_stderr(null), \
                    self.assertRaises(SystemExit) as cm:
                main(["run", "--corpus", str(root), "--out", str(out)])
            self.assertEqual(cm.exception.code, 2)


class TestJudgeCost(unittest.TestCase):
    """B13: judge calls were never costed."""

    def test_judge_usage_is_recorded_on_the_row(self):
        case = Case(name="c", conversions={"clean": "x"}, questions=(
            Question(id="q", question="?", gold="a summary", type="open"),))
        judge = FakeModelClient(lambda m, s, p: '{"verdict": "correct", "reason": "ok"}')
        rows = run_case(case, client=FakeModelClient(lambda m, s, p: "an answer"),
                        models=[STRONG], judge=judge, judge_model="claude-sonnet-4-6")
        self.assertEqual(rows[0].judge_model, "claude-sonnet-4-6")
        self.assertGreater(rows[0].judge_input_tokens, 0)

    def test_programmatic_grade_records_no_judge(self):
        case = Case(name="c", conversions={"clean": "x"}, questions=(
            Question(id="q", question="?", gold="1", type="numeric"),))
        rows = run_case(case, client=FakeModelClient(lambda m, s, p: "1"),
                        models=[STRONG], judge=FakeModelClient(lambda m, s, p: "x"))
        self.assertEqual((rows[0].judge_model, rows[0].judge_input_tokens), ("", 0))


class _StatusError(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


class TestDeterministicFailures(unittest.TestCase):
    """B16: only context overflow was classified; refusals graded as wrong."""

    def _case(self):
        return Case(name="c", conversions={"clean": "x"}, questions=(
            Question(id="q", question="?", gold="1", type="numeric"),))

    def _rows(self, responder):
        return run_case(self._case(), client=FakeModelClient(responder), models=[STRONG])

    def test_request_too_large_is_recorded(self):
        def big(m, s, p):
            raise _StatusError("Error code: 413 - request_too_large", 413)
        self.assertEqual(self._rows(big)[0].status, REQUEST_TOO_LARGE)

    def test_pdf_page_limit_is_recorded(self):
        def pages(m, s, p):
            raise _StatusError("Error code: 400 - A maximum of 100 PDF pages may be provided.", 400)
        self.assertEqual(self._rows(pages)[0].status, PDF_TOO_MANY_PAGES)

    def test_other_bad_requests_still_raise(self):
        def bad(m, s, p):
            raise _StatusError("Error code: 400 - messages: roles must alternate", 400)
        with self.assertRaises(_StatusError):
            self._rows(bad)

    def test_refusal_is_a_status_not_a_wrong_answer(self):
        row = self._rows(lambda m, s, p: ("1", "refusal"))[0]
        self.assertEqual(row.status, REFUSAL)
        self.assertEqual(row.score, 0.0)
        self.assertEqual(row.answer, "1")
        self.assertGreater(row.input_tokens, 0)


class TestCurrentRowsDirect(unittest.TestCase):
    def test_unknown_case_is_out_of_scope(self):
        case = Case(name="c", conversions={"clean": "x"}, questions=(
            Question(id="q", question="?", gold="1", type="numeric"),))
        row = Result(case="other", conversion="clean", model=STRONG, question_id="q",
                     question_type="numeric", correct=True, score=1.0, input_tokens=1,
                     output_tokens=1, answer="1", detail="")
        kept, counts = current_rows([row, replace(row, case="c")], [case])
        self.assertEqual(len(kept), 1)
        self.assertEqual(counts["out_of_scope"], 1)


if __name__ == "__main__":
    unittest.main()
