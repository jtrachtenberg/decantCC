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


class TestValidation(unittest.TestCase):
    """B20: what questions.schema.json says, enforced at load."""

    def parse(self, questions):
        from decant_eval.corpus import _parse_questions

        return _parse_questions(questions, "questions.json")

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate question id"):
            self.parse([{"id": "a", "question": "?", "gold": "x"},
                        {"id": "a", "question": "??", "gold": "y"}])

    def test_bad_golds_and_tolerance_rejected(self):
        for q in (
            {"id": "a", "question": "?", "gold": "about 5", "type": "numeric"},
            {"id": "a", "question": "?", "gold": "5", "type": "numeric", "tolerance": -1},
            {"id": "a", "question": "?", "gold": "x", "type": "set"},
            {"id": "a", "question": "?", "gold": [], "type": "ordered_list"},
            {"id": "a", "question": "?", "gold": ["x"], "type": "exact"},
        ):
            with self.subTest(q=q), self.assertRaises(ValueError):
                self.parse([q])

    def test_valid_golds_accepted(self):
        qs = self.parse([
            {"id": "n", "question": "?", "gold": "3,782,020", "type": "numeric"},
            {"id": "m", "question": "?", "gold": 5.4, "type": "numeric", "tolerance": 0.05},
            {"id": "s", "question": "?", "gold": ["a", "b"], "type": "set"},
        ])
        self.assertEqual(len(qs), 3)

    def _case(self, tmp, files):
        import json

        d = Path(tmp) / "c"
        (d / "conversions").mkdir(parents=True)
        (d / "questions.json").write_text(json.dumps(
            {"questions": [{"id": "q", "question": "?", "gold": "x"}]}))
        for name, text in files.items():
            (d / name).write_text(text)
        return d

    def test_stem_collision_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = self._case(tmp, {"conversions/decant.md": "a", "conversions/decant.txt": "b"})
            with self.assertRaisesRegex(ValueError, "both"):
                load_case(d)

    def test_uppercase_suffix_is_a_conversion(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = self._case(tmp, {"conversions/decant.MD": "a"})
            self.assertEqual(sorted(load_case(d).conversions), ["decant"])

    def test_several_source_files_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = self._case(tmp, {"conversions/a.md": "a", "source.pdf": "%PDF", "source.txt": "t"})
            with self.assertRaisesRegex(ValueError, "several source files"):
                load_case(d)

    def test_real_corpus_passes_validation(self):
        corpus = Path(__file__).resolve().parent.parent / "corpus"
        self.assertTrue(load_corpus(corpus))


if __name__ == "__main__":
    unittest.main()
