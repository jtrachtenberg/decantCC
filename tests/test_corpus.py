import unittest
from pathlib import Path

from decant_eval.corpus import load_case, load_corpus

# The synthetic e2e fixture lives with the tests, not in the real corpus —
# its clean/garbled arms would otherwise poison a real run's report (no case
# in common with the real conversion arms => nothing comparable).
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestCorpus(unittest.TestCase):
    def test_loads_sample_case(self):
        case = load_case(FIXTURES / "sample-invoice")
        self.assertEqual(case.name, "sample-invoice")
        self.assertEqual(len(case.questions), 4)
        self.assertIn("clean", case.conversions)
        self.assertIn("garbled", case.conversions)
        # A question of each type is present.
        self.assertEqual(
            {q.type for q in case.questions}, {"numeric", "exact", "set", "open"}
        )

    def test_load_corpus_finds_cases(self):
        cases = load_corpus(FIXTURES)
        self.assertTrue(any(c.name == "sample-invoice" for c in cases))


if __name__ == "__main__":
    unittest.main()
