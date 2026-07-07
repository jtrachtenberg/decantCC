"""Grader tests — the validity-critical core, all offline. Run:

    cd eval && python -m unittest discover tests
"""

import unittest

from decant_eval.corpus import Question
from decant_eval.grading import grade
from decant_eval.models import FakeModelClient


def q(qtype, gold, tol=0.0):
    return Question(id="x", question="?", gold=gold, type=qtype, tolerance=tol)


def judge_saying(text):
    """A judge that always returns `text`, regardless of the prompt."""
    return FakeModelClient(lambda m, s, p: text)


class TestNumeric(unittest.TestCase):
    def test_within_tolerance_and_formatting(self):
        self.assertTrue(grade(q("numeric", "1250.00", 0.01), "The total is $1,250.00.")[0])
        self.assertTrue(grade(q("numeric", 5.4, 0.05), "a 5.4-year increase")[0])

    def test_wrong_number_fails(self):
        self.assertFalse(grade(q("numeric", "1250.00", 0.01), "The total is $800.00.")[0])

    def test_no_number_fails(self):
        self.assertFalse(grade(q("numeric", "1250", 0), "NOT FOUND")[0])

    def test_takes_first_number_of_final_line(self):
        # Multi-line: only the committed final line counts, not a mid-answer aside.
        ans = "Let me check the rows.\nThe subtotal was 800.00.\nANSWER: 1250.00"
        self.assertTrue(grade(q("numeric", "1250.00", 0.01), ans)[0])


class TestExact(unittest.TestCase):
    def test_equality_and_normalization(self):
        self.assertTrue(grade(q("exact", "Acme Corp"), "Acme Corp")[0])
        self.assertTrue(grade(q("exact", "Acme Corp"), "acme  corp")[0])  # case/space
        self.assertTrue(grade(q("exact", "Acme Corp"), "Acme Corp.")[0])  # trailing punct

    def test_barely_longer_containment_ok(self):
        self.assertTrue(grade(q("exact", "Acme Corp"), "Acme Corp Inc")[0])

    def test_absent_fails(self):
        self.assertFalse(grade(q("exact", "Acme Corp"), "Globex Inc.")[0])

    def test_verbose_answer_routes_to_judge(self):
        # A wordy but correct answer is not auto-credited; a judge adjudicates it.
        ok, _, detail = grade(
            q("exact", "Acme Corp"), "The vendor is Acme Corp.",
            judge=judge_saying('{"verdict": "correct", "reason": "same vendor"}'),
        )
        self.assertTrue(ok)
        self.assertIn("judge", detail)
        # ...and with no judge, an over-long containment does not pass silently.
        self.assertFalse(grade(q("exact", "Acme Corp"), "The vendor is Acme Corp.")[0])


class TestSet(unittest.TestCase):
    def test_partial_credit_and_full(self):
        gold = ["widgets", "gaskets", "shipping"]
        ok, score, _ = grade(q("set", gold), "widgets and gaskets")
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 2 / 3)
        self.assertTrue(grade(q("set", gold), "widgets, gaskets, shipping")[0])


class TestOpenJudge(unittest.TestCase):
    def test_judge_verdict_parsed(self):
        judge = judge_saying('{"verdict": "correct", "reason": "matches"}')
        ok, score, detail = grade(
            q("open", "gold"), "candidate", judge=judge, judge_model="claude-opus-4-8"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        self.assertIn("correct", detail)

    def test_partial_is_half(self):
        ok, score, _ = grade(q("open", "gold"), "x", judge=judge_saying("verdict: partial"))
        self.assertFalse(ok)
        self.assertEqual(score, 0.5)

    def test_no_judge_skips(self):
        ok, score, _ = grade(q("open", "gold"), "x", judge=None)
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_judge_error_is_scored_failure(self):
        def boom(m, s, p):
            raise RuntimeError("503 overloaded")

        ok, score, detail = grade(q("open", "gold"), "x", judge=FakeModelClient(boom))
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        self.assertIn("judge error", detail)


class TestReviewRegressions(unittest.TestCase):
    """The six executed-proof failures from the Fable review (memo table). Each
    was silently scored 1.0 by the old graders; each must now score 0. Pinned
    with no judge so the programmatic verdict is deterministic."""

    def test_negated_entity(self):
        ok, score, _ = grade(q("exact", "Acme Corp"), "It is definitely not Acme Corp.")
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_hedged_entity(self):
        ok, score, _ = grade(
            q("exact", "Acme Corp"),
            "The vendor field is garbled, but Acme Corp appears somewhere.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_negated_number(self):
        ok, score, _ = grade(
            q("numeric", "1250.00", 0.01),
            "The total is 800.00, not 1250.00 as some rows suggest.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_substring_set_item(self):
        ok, score, _ = grade(q("set", ["ship"]), "We offer shipping items")
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_boundary_entity(self):
        ok, score, _ = grade(q("exact", "Acme Corp"), "Acme Corporation of Delaware")
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_judge_not_correct_is_incorrect(self):
        ok, score, detail = grade(
            q("open", "gold"), "x", judge=judge_saying("The candidate is not correct.")
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        self.assertIn("unparseable", detail)


if __name__ == "__main__":
    unittest.main()
