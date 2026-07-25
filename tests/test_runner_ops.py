"""Ops and metric-integrity tests for the fixes from the Fable review:
JSONL persistence + resume, the raw-PDF arena anchor, the memory-contamination
control arm, and the report's common-case / floor / missing-tier guards. Offline.
"""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from decant_eval.corpus import Case, Question
from decant_eval.models import FakeModelClient
from decant_eval.report import (
    billed_input_tokens, build_report, cost_summary, to_markdown, truncation_note,
)
from decant_eval.runner import (
    CONTEXT_OVERFLOW, CONTROL, RAW, Result, load_completed, regrade_rows, run_case,
    run_control, run_corpus,
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


def case_with_two_qs():
    return Case(name="c", questions=qs(), conversions={"clean": "x"})


def answerer(model, system, prompt):
    if "total" in prompt.lower():
        return "1250.00"
    if "vendor" in prompt.lower():
        return "Acme Corp"
    return "NOT FOUND"


OVERFLOW_MSG = "Error code: 400 - prompt is too long: 201055 tokens > 200000 maximum"


def overflow_for_weak(model, system, prompt):
    """Responder that fails the weak tier the way the API does when a document
    exceeds the context window; FakeModelClient propagates the raise."""
    if model == WEAK:
        raise RuntimeError(OVERFLOW_MSG)
    return answerer(model, system, prompt)


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
            self.assertIn((case.name, "clean", STRONG, "total", 0), done)

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


class TestContextOverflow(unittest.TestCase):
    """A document too large for a target model's context window scores 0 for
    that (conversion, model) instead of crashing the run — not fitting the
    weak reader is a transfer failure the arena must record — and the row
    carries a machine-readable status so the report can show it as a failed
    call rather than a graded wrong answer."""

    def test_overflow_scores_zero_with_status_and_run_continues(self):
        case = Case(name="c", questions=qs(),
                    conversions={"clean": "Total: 1250.00 USD\nVendor: Acme Corp"})
        rows = run_case(case, client=FakeModelClient(overflow_for_weak),
                        models=[STRONG, WEAK])
        by = {(r.model, r.question_id): r for r in rows}
        self.assertEqual(len(rows), 4)  # both models recorded for both questions
        strong_row = by[(STRONG, "total")]
        self.assertTrue(strong_row.correct)  # strong tier unaffected
        self.assertEqual(strong_row.status, "")
        weak_row = by[(WEAK, "total")]
        self.assertFalse(weak_row.correct)
        self.assertEqual(weak_row.score, 0.0)
        self.assertEqual(weak_row.input_tokens, 0)  # nothing was billed
        self.assertEqual(weak_row.answer, "")
        self.assertEqual(weak_row.status, CONTEXT_OVERFLOW)
        self.assertIn("context overflow", weak_row.detail)

    def test_status_survives_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = _load(make_case(tmp))
            path = Path(tmp) / "rows.jsonl"
            run_case(case, client=FakeModelClient(overflow_for_weak),
                     models=[STRONG, WEAK], jsonl_path=path)
            reloaded, done = load_completed(path)
            weak_rows = [r for r in reloaded if r.model == WEAK]
            self.assertTrue(weak_rows)
            self.assertTrue(all(r.status == CONTEXT_OVERFLOW for r in weak_rows))
            self.assertTrue(all(r.status == "" for r in reloaded if r.model == STRONG))
            # resume treats the failed rows as done — deterministic, no re-bill
            self.assertIn((case.name, "clean", WEAK, "total", 0), done)

    def test_pre_status_jsonl_loads_and_classifies_legacy_failures(self):
        # Audit trails written before the status field existed (the 2026-07
        # shipped runs): a graded row loads with status "", and a failure row —
        # recognizable by the detail the old catch wrote — re-derives its status.
        graded = {"case": "c", "conversion": "clean", "model": STRONG,
                  "question_id": "q1", "question_type": "exact", "correct": True,
                  "score": 1.0, "input_tokens": 120, "output_tokens": 5,
                  "answer": "Acme Corp", "detail": "exact match", "source": ""}
        failed = {"case": "c", "conversion": RAW, "model": WEAK,
                  "question_id": "q1", "question_type": "exact", "correct": False,
                  "score": 0.0, "input_tokens": 0, "output_tokens": 0,
                  "answer": "", "detail": f"context overflow: {OVERFLOW_MSG}",
                  "source": ""}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(graded) + "\n" + json.dumps(failed) + "\n",
                            encoding="utf-8")
            rows, done = load_completed(path)
            self.assertEqual(rows[0].status, "")
            self.assertEqual(rows[1].status, CONTEXT_OVERFLOW)
            # Pre-repeats rows carry no `repeat`; it defaults to 0 so their keys
            # still match the 5-tuple resume key.
            self.assertIn(("c", "clean", STRONG, "q1", 0), done)
            self.assertIn(("c", RAW, WEAK, "q1", 0), done)

    def test_other_errors_still_raise(self):
        # Transient failures must crash the run so --resume retries them
        # instead of freezing a permanent 0 into the audit trail.
        def boom(model, system, prompt):
            raise RuntimeError("connection reset")
        case = Case(name="c", questions=qs(), conversions={"clean": "x"})
        with self.assertRaises(RuntimeError):
            run_case(case, client=FakeModelClient(boom), models=[STRONG])


class TestTruncation(unittest.TestCase):
    """A model cut off at the output cap (stop_reason "max_tokens") must be
    distinguishable from one that answered wrongly — otherwise a budget
    artifact reads as a representation failure. Has never fired at the 512-token
    default, so this is a guard, not a fix for an observed problem."""

    def test_truncated_row_is_flagged_and_still_graded(self):
        # Answers correctly but is cut off; the row is graded AND marked.
        client = FakeModelClient(lambda m, s, p: ("1250.00", "max_tokens"))
        case = Case(name="c", questions=qs(), conversions={"clean": "x"})
        rows = run_case(case, client=client, models=[STRONG])
        by = {r.question_id: r for r in rows}
        self.assertTrue(by["total"].truncated)
        self.assertTrue(by["total"].correct)  # graded on what did come back
        self.assertIn("truncated (max_tokens)", by["total"].detail)

    def test_untruncated_row_is_not_flagged(self):
        case = Case(name="c", questions=qs(), conversions={"clean": "x"})
        rows = run_case(case, client=FakeModelClient(answerer), models=[STRONG])
        self.assertTrue(all(not r.truncated for r in rows))
        self.assertTrue(all("truncated" not in r.detail for r in rows))

    def test_context_overflow_is_not_reported_as_truncation(self):
        case = Case(name="c", questions=qs(),
                    conversions={"clean": "Total: 1250.00 USD\nVendor: Acme Corp"})
        rows = run_case(case, client=FakeModelClient(overflow_for_weak),
                        models=[STRONG, WEAK])
        self.assertTrue(all(not r.truncated for r in rows))

    def test_control_arm_flags_truncation_too(self):
        client = FakeModelClient(lambda m, s, p: ("Acme Corp", "max_tokens"))
        case = Case(name="c", questions=qs(), conversions={"clean": "x"})
        rows = run_control(case, client=client, models=[STRONG])
        self.assertTrue(all(r.truncated for r in rows))

    def test_resume_loads_rows_written_before_the_field_existed(self):
        # A pre-truncation-field JSONL row must still load on --resume.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rows.jsonl"
            p.write_text(json.dumps({
                "case": "c", "conversion": "clean", "model": STRONG,
                "question_id": "total", "question_type": "numeric", "correct": True,
                "score": 1.0, "input_tokens": 10, "output_tokens": 2,
                "answer": "1250.00", "detail": "",
            }) + "\n", encoding="utf-8")
            rows, done = load_completed(p)
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0].truncated)
            self.assertEqual(rows[0].repeat, 0)
            self.assertIn(("c", "clean", STRONG, "total", 0), done)


class TestRepeats(unittest.TestCase):
    """Repeats are the only variance control the arena has (temperature is
    rejected on the strong tier), so each sample must be its own row, keyed
    distinctly enough that --resume never conflates two samples of one cell."""

    def test_default_is_one_sample_per_cell(self):
        case = Case(name="c", questions=qs(), conversions={"clean": "x"})
        rows = run_case(case, client=FakeModelClient(answerer), models=[STRONG])
        self.assertEqual(len(rows), 2)  # 2 questions x 1 conversion x 1 model
        self.assertTrue(all(r.repeat == 0 for r in rows))

    def test_repeats_produce_one_row_per_sample(self):
        case = Case(name="c", questions=qs(), conversions={"clean": "x"})
        rows = run_case(case, client=FakeModelClient(answerer), models=[STRONG], repeats=3)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sorted(r.repeat for r in rows if r.question_id == "total"), [0, 1, 2])

    def test_samples_of_one_cell_have_distinct_keys(self):
        # Same (case, conversion, model, question) — only `repeat` separates
        # them. A key without it would mark the whole cell done after sample 0.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rows.jsonl"
            run_case(case_with_two_qs(), client=FakeModelClient(answerer),
                     models=[STRONG], repeats=3, jsonl_path=p)
            rows, done = load_completed(p)
            self.assertEqual(len(rows), 6)
            self.assertEqual(len(done), 6)  # no collisions

    def test_resume_continues_a_partial_repeat_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "rows.jsonl"
            run_case(case_with_two_qs(), client=FakeModelClient(answerer),
                     models=[STRONG], repeats=1, jsonl_path=p)
            _, done = load_completed(p)
            fresh = run_case(case_with_two_qs(), client=FakeModelClient(answerer),
                             models=[STRONG], repeats=3, jsonl_path=p, done=done)
            # Sample 0 of each question was already done; only 1 and 2 re-run.
            self.assertEqual(len(fresh), 4)
            self.assertTrue(all(r.repeat in (1, 2) for r in fresh))

    def test_report_averages_repeats(self):
        # One question answered right twice and wrong once -> 0.67, not 1.0.
        answers = iter(["1250.00", "1250.00", "wrong"])
        client = FakeModelClient(lambda m, s, p: next(answers))
        case = Case(name="c", questions=qs()[:1], conversions={"clean": "x"})
        rows = run_case(case, client=client, models=[STRONG], repeats=3)
        rep = build_report(rows, strong=STRONG, weak=WEAK)
        self.assertAlmostEqual(rep.scores[0].accuracy[STRONG], 2 / 3, places=2)


class TestRegrade(unittest.TestCase):
    """A grader fix must be applicable to answers already paid for — and must
    not silently downgrade rows whose verdict came from a judge that isn't
    available offline."""

    def case(self):
        return Case(name="c", conversions={"clean": "x"}, questions=(
            Question(id="num", question="?", gold="1.6", type="numeric"),
            Question(id="ex", question="?", gold="Acme Corp", type="exact"),
            Question(id="op", question="?", gold="a summary", type="open"),
        ))

    def row(self, qid, qtype, answer, score):
        return Result(case="c", conversion="clean", model=STRONG, question_id=qid,
                      question_type=qtype, correct=score == 1.0, score=score,
                      input_tokens=1, output_tokens=1, answer=answer, detail="")

    def test_numeric_row_picks_up_a_grader_fix(self):
        # Scored 0 under the old grader; the equation rule now credits it.
        rows = [self.row("num", "numeric", "22.19 - 20.59 = 1.60", 0.0)]
        new, changed, skipped = regrade_rows(rows, [self.case()])
        self.assertEqual(changed, 1)
        self.assertEqual(new[0].score, 1.0)
        self.assertEqual(skipped, 0)

    def test_judge_backed_rows_keep_their_verdict_without_a_judge(self):
        # An `exact` row that passed via the judge would regrade to 0 offline.
        # It must be skipped, not downgraded.
        rows = [self.row("ex", "exact", "It was Acme Corp, per the invoice header.", 1.0),
                self.row("op", "open", "A long prose answer.", 1.0)]
        new, changed, skipped = regrade_rows(rows, [self.case()])
        self.assertEqual(changed, 0)
        self.assertEqual(skipped, 2)
        self.assertTrue(all(r.score == 1.0 for r in new))

    def test_judge_supplied_regrades_the_skipped_types(self):
        rows = [self.row("ex", "exact", "It was Acme Corp, per the invoice header.", 1.0)]
        judge = FakeModelClient(lambda m, s, p: "incorrect")
        new, changed, skipped = regrade_rows(rows, [self.case()], judge=judge, judge_model="j")
        self.assertEqual(skipped, 0)
        self.assertEqual(changed, 1)
        self.assertEqual(new[0].score, 0.0)

    def test_failed_call_rows_are_left_alone(self):
        # There is no answer text to re-grade, and the failure is the finding.
        # Keyed on `status`, not the detail prose — legacy trails get their
        # status re-derived by load_completed (TestContextOverflow covers that).
        r = self.row("num", "numeric", "", 0.0)
        r = Result(**{**asdict(r), "status": CONTEXT_OVERFLOW,
                      "detail": "context overflow: prompt is too long"})
        new, changed, skipped = regrade_rows([r], [self.case()])
        self.assertEqual((changed, skipped), (0, 0))
        self.assertEqual(new[0].status, CONTEXT_OVERFLOW)
        self.assertIn("context overflow", new[0].detail)

    def test_unknown_question_is_left_alone(self):
        rows = [self.row("vanished", "numeric", "1.60", 0.0)]
        new, changed, skipped = regrade_rows(rows, [self.case()])
        self.assertEqual((changed, skipped), (0, 0))
        self.assertEqual(new[0].score, 0.0)

    def test_other_row_fields_survive_regrading(self):
        rows = [Result(case="c", conversion="clean", model=STRONG, question_id="num",
                       question_type="numeric", correct=False, score=0.0,
                       input_tokens=99, output_tokens=7, answer="= 1.60", detail="",
                       source="table-10", repeat=2, cache_read_tokens=50)]
        new, _, _ = regrade_rows(rows, [self.case()])
        self.assertEqual(new[0].source, "table-10")
        self.assertEqual(new[0].repeat, 2)
        self.assertEqual(new[0].input_tokens, 99)
        self.assertEqual(new[0].cache_read_tokens, 50)


class TestCostAccounting(unittest.TestCase):
    """input_tokens is the total the request covered; the cached portions inside
    it bill at a fraction of base rate. Pricing all of it at full rate — what
    the client used to force by folding the components together — overstates a
    cached run several fold."""

    def row(self, **kw):
        base = dict(case="c", conversion="decant", model=STRONG, question_id="q",
                    question_type="exact", correct=True, score=1.0,
                    input_tokens=0, output_tokens=0, answer="a", detail="")
        return Result(**{**base, **kw})

    def test_fully_uncached_bills_at_face_value(self):
        r = self.row(input_tokens=1000)
        self.assertEqual(billed_input_tokens(r), 1000)

    def test_cache_read_bills_at_a_tenth(self):
        # 1000 total, 900 of it a cache read -> 100 + 90 = 190
        r = self.row(input_tokens=1000, cache_read_tokens=900)
        self.assertAlmostEqual(billed_input_tokens(r), 190.0)

    def test_cache_write_bills_at_1_25x(self):
        r = self.row(input_tokens=1000, cache_creation_tokens=800)
        self.assertAlmostEqual(billed_input_tokens(r), 200 + 800 * 1.25)

    def test_components_never_exceed_the_total(self):
        # Defensive: a malformed row must not produce negative uncached tokens.
        r = self.row(input_tokens=100, cache_read_tokens=500)
        self.assertAlmostEqual(billed_input_tokens(r), 50.0)

    def test_summary_prices_each_model_and_totals(self):
        rows = [
            self.row(model=STRONG, input_tokens=1_000_000, output_tokens=1_000),
            self.row(model=WEAK, input_tokens=1_000_000, output_tokens=4_000),
        ]
        out = cost_summary(rows)
        self.assertIn("$5.03", out)   # Opus: 1M in @ $5 + 1k out @ $25/M = 5.025
        self.assertIn("$1.02", out)   # Haiku: 1M in @ $1 + 4k out @ $5/M = 1.02
        # Total sums the unrounded values (6.045), so it can differ by a cent
        # from adding up the rounded per-model cells. That's display rounding,
        # not a miscount — the total is the more accurate figure.
        self.assertIn("$6.04", out)

    def test_uninstrumented_rows_are_flagged_as_a_ceiling(self):
        # A row with input but no cache components predates the recording.
        out = cost_summary([self.row(input_tokens=1_000_000)])
        self.assertIn("WARNING", out)
        self.assertIn("ceiling", out)

    def test_instrumented_rows_are_not_flagged(self):
        out = cost_summary([self.row(input_tokens=1_000_000, cache_read_tokens=900_000)])
        self.assertNotIn("WARNING", out)

    def test_unknown_model_is_named_not_silently_zeroed(self):
        out = cost_summary([self.row(model="claude-future-9", input_tokens=1000)])
        self.assertIn("claude-future-9", out)
        self.assertIn("No price on file", out)


class TestTruncationNote(unittest.TestCase):
    def test_note_lists_truncated_rows(self):
        rows = [
            Result(case="c", conversion="decant", model=STRONG, question_id="q1",
                   question_type="exact", correct=False, score=0.0, input_tokens=1,
                   output_tokens=1, answer="", detail="", truncated=True),
            Result(case="c", conversion="decant", model=STRONG, question_id="q2",
                   question_type="exact", correct=True, score=1.0, input_tokens=1,
                   output_tokens=1, answer="x", detail=""),
        ]
        note = truncation_note(rows)
        self.assertIn("1 of 2", note)
        self.assertIn("q1", note)
        self.assertNotIn("q2", note)

    def test_no_note_when_nothing_truncated(self):
        rows = [Result(case="c", conversion="decant", model=STRONG, question_id="q2",
                       question_type="exact", correct=True, score=1.0, input_tokens=1,
                       output_tokens=1, answer="x", detail="")]
        self.assertEqual(truncation_note(rows), "")


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


class TestFailedCallAnnotation(unittest.TestCase):
    """Failed API calls must surface in the scoreboard as failed calls, not as
    an arm that answered everything wrong — the shipped messy-scan report read
    'raw: Haiku 0.00 / spread +0.70' when the PDF simply didn't fit Haiku."""

    def _row(self, conv, model, qid, score, status=""):
        failed = bool(status)
        return Result(
            "c1", conv, model, qid, "exact", score == 1.0, score,
            0 if failed else 100, 0 if failed else 5, "" if failed else "a",
            f"context overflow: {OVERFLOW_MSG}" if failed else "d", status=status,
        )

    def _rows_with_weak_overflow_on_raw(self):
        rows = []
        for i in range(4):
            rows += [
                self._row(RAW, STRONG, f"q{i}", 1.0),
                self._row(RAW, WEAK, f"q{i}", 0.0, status=CONTEXT_OVERFLOW),
                self._row("decant", STRONG, f"q{i}", 1.0),
                self._row("decant", WEAK, f"q{i}", 1.0),
            ]
        return rows

    def test_failed_calls_counted_per_arm(self):
        rep = build_report(self._rows_with_weak_overflow_on_raw(), strong=STRONG, weak=WEAK)
        raw = next(cs for cs in rep.scores if cs.conversion == RAW)
        self.assertEqual(raw.failed_calls[WEAK], (4, 4, (CONTEXT_OVERFLOW,)))
        self.assertNotIn(STRONG, raw.failed_calls)  # strong tier answered fine
        decant = next(cs for cs in rep.scores if cs.conversion == "decant")
        self.assertEqual(decant.failed_calls, {})

    def test_markdown_flags_cells_and_footnotes_failed_arm(self):
        md = to_markdown(build_report(self._rows_with_weak_overflow_on_raw(),
                                      strong=STRONG, weak=WEAK))
        self.assertIn("0.00!", md)   # not a bare 0.00 accuracy
        self.assertIn("+1.00!", md)  # spread built on the failed tier is flagged too
        self.assertIn(f"raw / {WEAK}: 4/4 calls failed", md)
        self.assertIn("does not fit the model's context window", md)
        self.assertTrue(md.isascii(), "report must survive a cp1252 Windows console")

    def test_clean_report_carries_no_failure_marks(self):
        rows = [self._row("decant", STRONG, "q", 1.0), self._row("decant", WEAK, "q", 1.0)]
        md = to_markdown(build_report(rows, strong=STRONG, weak=WEAK))
        self.assertNotIn("calls failed", md)
        self.assertNotIn("!", md)


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
            self.assertIn(("c", "clean", STRONG, "q", 0), done)

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
