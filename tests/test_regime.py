"""The measurement regime is recorded on every row and asserted in the report.

Until Opus 5, `model` pinned the regime by itself: every model the harness ran
thought nothing when `thinking` was omitted, which is all answer() ever sends.
That stopped being true, silently -- same request, same code, different
experiment. These tests cover the three places the harness now refuses to let
that happen quietly: the model allowlist (cli), the regime fields on the row
(runner), and the sections that surface them (report).
"""

import contextlib
import os
import unittest
from dataclasses import replace
from pathlib import Path

from decant_eval.cli import main
from decant_eval.corpus import load_case
from decant_eval.grading import JUDGE_ERROR_PREFIX, _grade_open
from decant_eval.models import (
    THINKING_OFF_BY_DEFAULT, THINKING_ON_BY_DEFAULT, AnswerResult, FakeModelClient,
    default_thinks,
)
from decant_eval.report import judge_failure_note, regime_note
from decant_eval.runner import Result, run_case

FIXTURES = Path(__file__).resolve().parent / "fixtures"
STRONG, WEAK = "claude-opus-4-8", "claude-haiku-4-5"


_BASE = Result(
    case="c", conversion="decant", model=STRONG, question_id="q",
    question_type="numeric", correct=True, score=1.0, input_tokens=100,
    output_tokens=5, answer="1", detail="",
)


def _row(**kw) -> Result:
    return replace(_BASE, **kw)


class TestModelClassification(unittest.TestCase):
    def test_sets_are_disjoint(self):
        self.assertEqual(THINKING_OFF_BY_DEFAULT & THINKING_ON_BY_DEFAULT, frozenset())

    def test_known_defaults(self):
        # The whole allowlist rests on these being right; see models.py.
        self.assertIs(default_thinks("claude-opus-4-8"), False)
        self.assertIs(default_thinks("claude-haiku-4-5"), False)
        self.assertIs(default_thinks("claude-opus-5"), True)
        self.assertIs(default_thinks("claude-sonnet-5"), True)

    def test_unknown_model_is_none_not_false(self):
        # None must not collapse to False: an unclassified model is an open
        # question, and assuming it is safe is how this became a problem.
        self.assertIsNone(default_thinks("claude-opus-9"))
        self.assertIsNot(default_thinks("claude-opus-9"), False)


class TestAllowlist(unittest.TestCase):
    """argparse errors exit 2; each case must be refused before anything bills."""

    def _run(self, argv):
        # argparse writes usage to stderr on error; swallow it so a passing
        # suite stays readable.
        with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
            with self.assertRaises(SystemExit) as cm:
                main(argv)
        return cm.exception.code

    def _base(self, *extra):
        return ["run", "--corpus", str(FIXTURES), "--out", "unused.md", *extra]

    def test_thinking_model_refused_on_each_flag(self):
        for flag in ("--strong", "--weak", "--judge"):
            with self.subTest(flag=flag):
                self.assertEqual(self._run(self._base(flag, "claude-opus-5")), 2)

    def test_unclassified_model_refused(self):
        self.assertEqual(self._run(self._base("--strong", "claude-opus-9")), 2)

    def test_regrade_judge_is_validated(self):
        code = self._run(["regrade", "--corpus", str(FIXTURES),
                          "--rows", "nope.jsonl", "--judge", "claude-opus-5"])
        self.assertEqual(code, 2)

    def test_weak_only_ignores_unused_strong(self):
        # --strong is never called under --weak-only, so validating it would
        # refuse a run that cannot be affected. Reaching the missing corpus
        # proves validation passed.
        with self.assertRaises((SystemExit, FileNotFoundError)) as cm:
            main(["run", "--corpus", "no-such-dir", "--out", "unused.md",
                  "--weak-only", "--strong", "claude-opus-5"])
        self.assertNotIsInstance(cm.exception, SystemExit)

    def test_escape_hatch_allows_thinking_model(self):
        # Deliberate opt-in gets past validation (and is still flagged in the
        # report, which regime_note covers separately).
        with self.assertRaises((SystemExit, FileNotFoundError)) as cm:
            main(["run", "--corpus", "no-such-dir", "--out", "unused.md",
                  "--allow-thinking-default", "--strong", "claude-opus-5"])
        self.assertNotIsInstance(cm.exception, SystemExit)

    def test_escape_hatch_does_not_excuse_unknown_model(self):
        # Opting into thinking is a claim about a model you understand; it says
        # nothing about one nobody classified.
        self.assertEqual(
            self._run(self._base("--allow-thinking-default", "--strong", "claude-opus-9")), 2
        )


class TestRegimeRecorded(unittest.TestCase):
    def test_run_case_stamps_regime_on_every_row(self):
        case = load_case(FIXTURES / "sample-invoice")
        client = FakeModelClient(lambda m, s, p: "1250.00 USD")
        rows = run_case(case, client=client, models=[STRONG], max_tokens=777)
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r.max_tokens, 777)
            self.assertEqual(r.effort, "")      # not sent -> model's own default
            self.assertEqual(r.thinking, "")

    def test_legacy_rows_default_and_are_distinguishable(self):
        # max_tokens 0 is the "predates the field" marker, not a real budget.
        self.assertEqual(_row().max_tokens, 0)
        self.assertNotEqual(_row(max_tokens=512).max_tokens, 0)

    def test_answer_result_carries_regime_back(self):
        self.assertEqual(AnswerResult(text="x", input_tokens=1, output_tokens=1).effort, "")


class TestRegimeNote(unittest.TestCase):
    def test_silent_when_uniform_and_safe(self):
        md = regime_note([_row(max_tokens=512), _row(max_tokens=512)])
        self.assertNotIn("WARNING", md)

    def test_flags_mixed_regime_for_one_model(self):
        md = regime_note([_row(max_tokens=512), _row(max_tokens=1024)])
        self.assertIn("MORE THAN ONE regime", md)
        self.assertIn("arm boundary", md)

    def test_effort_difference_alone_counts_as_mixed(self):
        md = regime_note([_row(max_tokens=512), _row(max_tokens=512, effort="high")])
        self.assertIn("MORE THAN ONE regime", md)

    def test_same_regime_across_two_models_is_not_mixed(self):
        md = regime_note([_row(max_tokens=512), _row(max_tokens=512, model=WEAK)])
        self.assertNotIn("MORE THAN ONE regime", md)

    def test_flags_uninstrumented_rows(self):
        md = regime_note([_row(), _row()])
        self.assertIn("predate regime recording", md)
        self.assertIn("2 row(s)", md)

    def test_flags_thinking_model(self):
        md = regime_note([_row(model="claude-opus-5", max_tokens=512)])
        self.assertIn("adaptive thinking", md)
        self.assertIn("runs adaptive", md)  # singular

    def test_flags_unclassified_model(self):
        md = regime_note([_row(model="claude-opus-9", max_tokens=512)])
        self.assertIn("no thinking-default on file", md)

    def test_empty_rows_render_nothing(self):
        self.assertEqual(regime_note([]), "")


class TestJudgeFailure(unittest.TestCase):
    def test_truncated_verdict_is_not_a_graded_zero(self):
        # A judge cut off mid-JSON leaves _parse_verdict nothing to parse, and
        # its no-guessing rule returns `incorrect` -- a budget artifact scored
        # as a wrong answer. It must be reported as a judge failure instead.
        class Truncating:
            def answer(self, **kw):
                return AnswerResult(text='{"verdict": "corr',
                                    input_tokens=1, output_tokens=1,
                                    stop_reason="max_tokens")

        correct, score, detail = _grade_open("a", "b", "q?", Truncating(), STRONG)
        self.assertFalse(correct)
        self.assertIn(JUDGE_ERROR_PREFIX, detail)
        self.assertIn("truncated", detail)

    def test_complete_verdict_still_grades(self):
        judge = FakeModelClient(lambda m, s, p: '{"verdict": "correct", "reason": "ok"}')
        correct, score, detail = _grade_open("a", "b", "q?", judge, STRONG)
        self.assertTrue(correct)
        self.assertNotIn(JUDGE_ERROR_PREFIX, detail)

    def test_judge_outage_still_flagged(self):
        class Broken:
            def answer(self, **kw):
                raise RuntimeError("503")

        _, _, detail = _grade_open("a", "b", "q?", Broken(), STRONG)
        self.assertIn(JUDGE_ERROR_PREFIX, detail)

    def test_note_lists_failures_and_is_silent_otherwise(self):
        clean = [_row(detail="answer 1.0 within +/-0.0 of 1.0")]
        self.assertEqual(judge_failure_note(clean), "")
        bad = clean + [_row(score=0.0, correct=False,
                            detail=f"{JUDGE_ERROR_PREFIX} verdict truncated")]
        md = judge_failure_note(bad)
        self.assertIn("Judge failures", md)
        self.assertIn("1 of 2 rows", md)

    def test_note_survives_the_truncation_prefix(self):
        # runner prepends "truncated (max_tokens): " when the ANSWER was cut
        # off, so the judge marker is not at the start of the string.
        md = judge_failure_note(
            [_row(detail=f"truncated (max_tokens): {JUDGE_ERROR_PREFIX} boom")]
        )
        self.assertIn("Judge failures", md)


if __name__ == "__main__":
    unittest.main()
