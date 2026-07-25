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


class TestOrderedList(unittest.TestCase):
    GOLD = ["Identify", "Assess", "Treat", "Report", "Monitor"]

    def test_full_credit_in_order(self):
        ans = "The steps are: identify, assess, treat, report, and monitor."
        ok, score, _ = grade(q("ordered_list", self.GOLD), ans)
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_wrong_order_penalized(self):
        # "assess" before "identify": identify matches at its later position is
        # impossible, so everything from the swap scores as misses in sequence.
        ans = "assess, identify, treat, report, monitor"
        ok, score, _ = grade(q("ordered_list", self.GOLD), ans)
        self.assertFalse(ok)
        self.assertLess(score, 1.0)

    def test_missing_item_partial(self):
        ok, score, _ = grade(q("ordered_list", self.GOLD), "identify, assess, treat")
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 3 / 5)

    def test_substring_not_matched(self):
        ok, score, _ = grade(q("ordered_list", ["treat"]), "treatment plans")
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)


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


class TestVerboseNumericRegressions(unittest.TestCase):
    """Two real rows from the 2026-07-15 billed run scored 0 despite containing
    the exact gold value: the committed number was buried behind a lead-in year
    or above a citation line, so _numbers()[0] grabbed the wrong figure. The
    grader now prefers a bolded number. The negation guard must survive: a value
    the model bolds is committed, but "800.00, not 1250.00" (no bold) still fails.
    """

    def test_bolded_value_beats_leadin_year(self):
        # Row 1: gold 808, but the sentence opens with the year 2025.
        ans = (
            "According to the CERN staff breakdown for 2025, CERN employed "
            "**808 technicians**, which represented 29.25% of personnel."
        )
        self.assertTrue(grade(q("numeric", "808", 0), ans)[0])

    def test_bolded_value_beats_citation_line(self):
        # Row 2: gold 42, bolded above a last line whose first number is 1.
        ans = (
            "Internal Revenue Code **Section 42** governs the credit.\n\n"
            'This is stated in NOTE 1 - ORGANIZATION: "...Section 42..."'
        )
        self.assertTrue(grade(q("numeric", "42", 0), ans)[0])

    def test_negation_without_bold_still_fails(self):
        # No bold -> fall back to the short answer's first number (800), not gold.
        ok, score, _ = grade(
            q("numeric", "1250.00", 0.01),
            "The total is 800.00, not 1250.00 as some rows suggest.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)


class TestShownArithmetic(unittest.TestCase):
    """messy-scan/well-depth-difference: raw/Opus answered the correct value
    1.60 in BOTH 2026-07-24 runs and scored 0 both times, because an answer that
    shows its work leads with an input, not the result. The commitment is what
    follows the last `=` or `:`."""

    def test_equation_result_beats_left_operand(self):
        # The AFTER-run row: correct value 1.60, previously graded 22.19.
        ans = "From Table 10, DTX1: Top = 20.59, Bottom = 22.19.\n\n22.19 - 20.59 = 1.60"
        ok, _, detail = grade(q("numeric", "1.6", 0), ans)
        self.assertTrue(ok, detail)

    def test_labelled_result_beats_leading_inputs(self):
        # The BEFORE-run row, one line, no equals sign: previously graded 20.59.
        ans = "Top: 20.59, Bottom: 22.19. Difference: 1.60"
        ok, _, detail = grade(q("numeric", "1.6", 0), ans)
        self.assertTrue(ok, detail)

    def test_unicode_minus_equation(self):
        # Models emit U+2212, not ASCII hyphen — the real row did.
        ans = "22.19 − 20.59 = 1.60"
        self.assertTrue(grade(q("numeric", "1.6", 0), ans)[0])

    def test_last_equation_wins(self):
        self.assertTrue(grade(q("numeric", "7", 0), "x = 5, then y = 7")[0])

    def test_bold_still_outranks_the_equation(self):
        # Bold is the stronger commitment signal and must keep precedence.
        self.assertTrue(grade(q("numeric", "42", 0), "**42** is the answer, since 1 = 1")[0])

    def test_plain_answer_unaffected(self):
        self.assertTrue(grade(q("numeric", "1.6", 0), "1.60")[0])

    def test_negation_guard_survives(self):
        # No "=" and no bold -> still the short answer's first number.
        ok, _, _ = grade(q("numeric", "1250.00", 0.01), "The total is 800.00, not 1250.00.")
        self.assertFalse(ok)


class TestSpelledOutNumbers(unittest.TestCase):
    """clean-text/land-improvements-life scored 0 on EVERY arm in the
    2026-07-15/16 billed runs: the source spells the value out ("fifteen years
    for land improvements"), models quoted it verbatim, and _NUM found no digit.
    Word-numbers are a fallback only — digits always take precedence."""

    def test_verbatim_quote_word_number(self):
        # Opus's actual committed answer from the billed run.
        self.assertTrue(grade(q("numeric", "15", 0), "fifteen years")[0])
        self.assertTrue(grade(q("numeric", "15", 0), "Fifteen years.")[0])

    def test_wrong_word_number_fails(self):
        self.assertFalse(grade(q("numeric", "15", 0), "forty years")[0])

    def test_compound_word_numbers(self):
        self.assertTrue(grade(q("numeric", "21", 0), "twenty-one units")[0])
        self.assertTrue(grade(q("numeric", "40", 0), "forty")[0])
        self.assertTrue(grade(q("numeric", "305", 0), "three hundred and five")[0])

    def test_digits_still_win_over_words(self):
        # "not fifteen" must not rescue a wrong digit answer.
        self.assertFalse(grade(q("numeric", "15", 0), "40 years, not fifteen")[0])

    def test_no_number_still_fails(self):
        # NOT FOUND is now reported as a decline rather than as a parse miss;
        # both are failures, the decline is just the more accurate reason.
        ok, _, detail = grade(q("numeric", "15", 0), "NOT FOUND")
        self.assertFalse(ok)
        self.assertIn("declined", detail)
        # An answer with no number and no decline still reports the parse miss.
        ok, _, detail = grade(q("numeric", "15", 0), "The document is silent on this.")
        self.assertFalse(ok)
        self.assertIn("no number", detail)


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


class TestContextNumbers(unittest.TestCase):
    """A prose answer usually restates the question's context before the value,
    so the *first* number is a year or a section label rather than the answer.
    22 of 304 numeric rows in the 2026-07 trails were graded against one of
    those, and every single one was a weak-tier row -- the defect inflated the
    reliability spread, which is the harness's headline metric."""

    def test_leading_year_is_not_the_answer(self):
        self.assertTrue(grade(q("numeric", "70"), (
            "According to the document, in 1999, the three MWRD properties "
            "encompassed almost 70 square miles of farmland."))[0])

    def test_for_year_is_not_the_answer(self):
        self.assertTrue(grade(q("numeric", "808"), (
            "According to the CERN staff breakdown for 2025, CERN employed 808 "
            "technicians, which represented 29.25% of the total staff."))[0])

    def test_section_label_is_not_the_answer(self):
        self.assertTrue(grade(q("numeric", "3782020"), (
            "MSIM's total financed Scope 1 GHG emissions for 2025 were "
            "3,782,020 tCO2e."))[0])

    def test_year_gold_still_grades_when_it_is_the_only_number(self):
        # Discounting context numbers must not make a year-valued answer
        # ungradeable: with nothing else to pick, the year stands.
        self.assertTrue(grade(q("numeric", "2021"), "The report was published in 2021.")[0])

    def test_date_word_does_not_discount_a_non_year(self):
        self.assertTrue(grade(q("numeric", "70"), "Biosolids are applied for 70 days.")[0])

    def test_negation_guard_survives_context_filtering(self):
        self.assertFalse(grade(q("numeric", "1250.00", 0.01), "800.00, not 1250.00")[0])


class TestRatioColon(unittest.TestCase):
    """The commitment-marker rule reads the text after the last `=` or `:` as
    the model's result. A colon *between digits* is a ratio, not a marker."""

    def test_ratio_answer_grades_as_the_ratio_value(self):
        self.assertTrue(grade(q("numeric", "2.31", 0.001), "2.31, or 2.31:1")[0])

    def test_label_colon_is_still_a_marker(self):
        self.assertTrue(grade(q("numeric", "1.60", 0.001),
                              "Top: 20.59, Bottom: 22.19. Difference: 1.60")[0])


class TestDeclinedAnswers(unittest.TestCase):
    """ANSWER_SYSTEM tells the model to reply exactly NOT FOUND when the
    document lacks the answer. When it does, the explanation that follows is an
    account of what it could not find -- mining it for a match credits the
    hedged non-answer a corrupted conversion provokes."""

    def test_not_found_then_explanation_is_not_credited(self):
        correct, score, detail = grade(q("numeric", "75"), (
            "NOT FOUND\n\nThe document does not contain the distance. It states "
            'that "biosolids are transported about 75 mi east from Denver", but '
            "that is a different measurement."))
        self.assertFalse(correct)
        self.assertEqual(score, 0.0)
        self.assertIn("declined", detail)

    def test_declined_set_answer_is_not_credited(self):
        correct, score, _ = grade(q("set", ["medium", "long term"]), (
            "NOT FOUND\n\nThe document does not specify a time frame. It only "
            'lists "MT LT" (indicating Medium Term and Long Term) without a '
            "designation."))
        self.assertFalse(correct)
        self.assertEqual(score, 0.0)

    def test_plain_not_found_still_fails(self):
        self.assertFalse(grade(q("numeric", "75"), "NOT FOUND")[0])

    def test_answer_merely_mentioning_not_found_is_still_graded(self):
        # The rule keys on the opening line, so a real answer that happens to
        # discuss the phrase is unaffected.
        self.assertTrue(grade(q("numeric", "75"), (
            "The distance is 75 miles; nothing here is NOT FOUND."))[0])


if __name__ == "__main__":
    unittest.main()
