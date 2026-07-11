"""End-to-end offline test: run the arena over the sample case with a scripted
model, and confirm the report separates the clean conversion from the garbled
one — including a wider reliability spread on the garbled table for the weak
model, the harness's whole reason to exist.
"""

import unittest
from pathlib import Path

from decant_eval.corpus import load_case
from decant_eval.models import FakeModelClient
from decant_eval.report import build_report, to_markdown
from decant_eval.runner import run_case

FIXTURES = Path(__file__).resolve().parent / "fixtures"
STRONG, WEAK = "claude-opus-4-8", "claude-haiku-4-5"


def scripted(model, system, prompt):
    """Fake reader: answers correctly from the clean conversion; from the
    garbled one the strong model recovers a bit, the weak model fails."""
    clean = "Total: 1250.00 USD" in prompt
    q = prompt.rsplit("QUESTION:", 1)[-1].lower()

    def ans(correct):
        if "total" in q:
            return "1250.00 USD" if correct else "800.00 USD"
        if "vendor" in q:
            return "Acme Corp" if correct else "unknown"
        if "line items" in q:
            return "widgets, gaskets, shipping" if correct else "widgets"
        if "one sentence" in q:
            return (
                "An invoice from Acme Corp billing 1250.00 USD for widgets, gaskets, and shipping."
                if correct
                else "Some invoice with jumbled amounts."
            )
        return "NOT FOUND"

    if clean:
        return ans(True)
    # Garbled: strong model half-right, weak model wrong.
    return ans(model == STRONG and "total" not in q)


class TestRunnerReport(unittest.TestCase):
    def setUp(self):
        case = load_case(FIXTURES / "sample-invoice")
        client = FakeModelClient(scripted)
        # Judge scores the open question by keyword in the candidate.
        judge = FakeModelClient(
            lambda m, s, p: "correct" if "Acme Corp billing 1250.00" in p else "incorrect"
        )
        self.rows = run_case(
            case, client=client, models=[STRONG, WEAK], judge=judge, judge_model=STRONG
        )
        self.report = build_report(self.rows, strong=STRONG, weak=WEAK)

    def test_every_combination_ran(self):
        # 2 conversions × 2 models × 4 questions
        self.assertEqual(len(self.rows), 16)

    def test_clean_beats_garbled_and_ranks_first(self):
        by_conv = {cs.conversion: cs for cs in self.report.scores}
        clean, garbled = by_conv["clean"], by_conv["garbled"]
        self.assertEqual(clean.accuracy[STRONG], 1.0)
        self.assertEqual(clean.accuracy[WEAK], 1.0)
        self.assertLess(garbled.accuracy[WEAK], clean.accuracy[WEAK])
        self.assertEqual(self.report.scores[0].conversion, "clean")  # ranked first

    def test_reliability_spread_wider_on_garbled(self):
        by_conv = {cs.conversion: cs for cs in self.report.scores}
        # Clean transfers to both tiers equally (spread ~0); garbled favors the
        # strong reader, so its spread is larger.
        self.assertEqual(by_conv["clean"].spread, 0.0)
        self.assertGreater(by_conv["garbled"].spread, by_conv["clean"].spread)

    def test_markdown_renders(self):
        md = to_markdown(self.report)
        self.assertIn("| conversion |", md)
        self.assertIn("clean", md)
        self.assertIn("spread", md)


if __name__ == "__main__":
    unittest.main()
