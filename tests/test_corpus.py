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

    def test_source_tag_parsed_and_optional(self):
        from decant_eval.corpus import _parse_questions

        qs = _parse_questions(
            [
                {"id": "a", "question": "?", "gold": "x", "source": "figure-12"},
                {"id": "b", "question": "?", "gold": "y"},
            ],
            "test",
        )
        self.assertEqual(qs[0].source, "figure-12")
        self.assertEqual(qs[1].source, "")  # untagged is the default, not an error


class TestSplits(unittest.TestCase):
    """Question banks. Iterating a compressor against the same questions it is
    reported on converges on keeping only the asked-about facts, so `dev` is
    what you tune against and `test` is what makes a declared win credible."""

    def bank(self):
        from decant_eval.corpus import _parse_questions

        return _parse_questions(
            [
                {"id": "d1", "question": "?", "gold": "x", "split": "dev"},
                {"id": "d2", "question": "?", "gold": "x", "split": "dev"},
                {"id": "t1", "question": "?", "gold": "x", "split": "test"},
                {"id": "r1", "question": "?", "gold": "x", "split": "retired"},
                {"id": "u1", "question": "?", "gold": "x"},
            ],
            "test",
        )

    def test_split_parsed_and_optional(self):
        qs = self.bank()
        self.assertEqual(qs[0].split, "dev")
        self.assertEqual(qs[4].split, "")  # unassigned is the default, not an error

    def test_unknown_split_is_an_error(self):
        from decant_eval.corpus import _parse_questions

        with self.assertRaises(ValueError):
            _parse_questions([{"id": "a", "question": "?", "gold": "x", "split": "dve"}], "t")

    def test_all_excludes_retired_only(self):
        from decant_eval.corpus import select_split

        # Retiring a question has to actually stop it costing money, or the
        # label means nothing — but an unassigned question still runs.
        ids = [q.id for q in select_split(self.bank(), "all")]
        self.assertEqual(ids, ["d1", "d2", "t1", "u1"])

    def test_dev_and_test_select_exactly_their_bank(self):
        from decant_eval.corpus import select_split

        self.assertEqual([q.id for q in select_split(self.bank(), "dev")], ["d1", "d2"])
        self.assertEqual([q.id for q in select_split(self.bank(), "test")], ["t1"])
        self.assertEqual([q.id for q in select_split(self.bank(), "retired")], ["r1"])

    def test_untagged_corpus_is_unaffected_by_the_default(self):
        # Every existing case is untagged; the default must load them all.
        case = load_case(FIXTURES / "sample-invoice")
        self.assertEqual(len(case.questions), 4)

    def test_case_with_no_questions_in_the_bank_is_dropped(self):
        # `--split dev` on a corpus where only some cases have a dev bank runs
        # those cases rather than failing on the ones that don't.
        with self.assertRaises(ValueError):
            load_corpus(FIXTURES, split="dev")


if __name__ == "__main__":
    unittest.main()
