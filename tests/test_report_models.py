"""Scoreboard, pricing and model-table fixes from the 2026-09 review (B1, B11,
B15, B17, B18, B21, B24, O2, O8). Offline."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from decant_eval.cli import main
from decant_eval.models import (
    AnthropicModelClient, FakeModelClient, base_model, default_thinks,
)
from decant_eval.report import (
    billed_input_tokens, build_report, cost_summary, regime_note, to_markdown,
)
from decant_eval.runner import CONTEXT_OVERFLOW, Result

STRONG, WEAK = "claude-opus-4-8", "claude-haiku-4-5"

_BASE = Result(case="c", conversion="a", model=STRONG, question_id="q",
               question_type="numeric", correct=True, score=1.0, input_tokens=100,
               output_tokens=5, answer="1", detail="", max_tokens=512)


def row(**kw):
    return replace(_BASE, **kw)


class TestCoverage(unittest.TestCase):
    """B1: one partial arm collapsed the scoreboard to a single case."""

    def _rows(self):
        rows = []
        for case in ("c1", "c2", "c3"):
            rows.append(row(case=case, conversion="decant"))
        rows.append(row(case="c1", conversion="ocr"))
        return rows

    def test_minority_coverage_is_a_headline_warning(self):
        rep = build_report(self._rows(), strong=STRONG, weak=WEAK)
        self.assertEqual(rep.partial_conversions, {"ocr": 1})
        md = to_markdown(rep)
        self.assertIn("WARNING: this table is scored on 1 of 3 cases", md)
        self.assertIn("ocr (1/3 cases)", md)
        self.assertLess(md.index("WARNING"), md.index("| conversion"))

    def _corpus(self, tmp):
        for case in ("c1", "c2", "c3"):
            d = Path(tmp) / "corp" / case
            (d / "conversions").mkdir(parents=True)
            (d / "questions.json").write_text(json.dumps({"questions": [
                {"id": "q", "question": "total?", "gold": "1", "type": "numeric"}]}))
            (d / "conversions" / "decant.md").write_text("1")
        (Path(tmp) / "corp" / "c1" / "conversions" / "ocr.md").write_text("1")
        return Path(tmp) / "corp"

    def test_run_is_refused_before_billing(self):
        with tempfile.TemporaryDirectory() as tmp:
            corp = self._corpus(tmp)
            fake = FakeModelClient(lambda m, s, p: "1")
            with mock.patch("decant_eval.cli.AnthropicModelClient", return_value=fake), \
                    open(os.devnull, "w") as null, contextlib.redirect_stderr(null), \
                    self.assertRaises(SystemExit) as cm:
                main(["run", "--corpus", str(corp), "--out", str(Path(tmp) / "r.md")])
            self.assertEqual(cm.exception.code, 2)
            self.assertEqual(fake.calls, [])

    def test_exclude_arm_scores_every_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            corp = self._corpus(tmp)
            fake = FakeModelClient(lambda m, s, p: "1")
            out = Path(tmp) / "r.md"
            with mock.patch("decant_eval.cli.AnthropicModelClient", return_value=fake), \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["run", "--corpus", str(corp), "--out", str(out),
                      "--exclude-arm", "ocr", "--no-control"])
            md = out.read_text()
            self.assertNotIn("ocr", md.split("## ")[0])
            self.assertIn("| decant | 1.00 | 1.00 | ", md)
            self.assertNotIn("common case(s); excluded", md)

    def test_allow_partial_arms_runs_anyway(self):
        with tempfile.TemporaryDirectory() as tmp:
            corp = self._corpus(tmp)
            fake = FakeModelClient(lambda m, s, p: "1")
            with mock.patch("decant_eval.cli.AnthropicModelClient", return_value=fake), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["run", "--corpus", str(corp), "--no-control",
                                       "--out", str(Path(tmp) / "r.md"),
                                       "--allow-partial-arms"]), 0)


class TestScoreboardEdges(unittest.TestCase):
    def test_failed_calls_do_not_lighten_the_cost_column(self):
        # B11: 200K arm with 2 of 3 calls overflowing averaged to 66,667.
        rows = [row(input_tokens=200_000, question_id="q0"),
                row(input_tokens=0, question_id="q1", status=CONTEXT_OVERFLOW, score=0.0),
                row(input_tokens=0, question_id="q2", status=CONTEXT_OVERFLOW, score=0.0)]
        rep = build_report(rows, strong=STRONG, weak=WEAK)
        self.assertEqual(rep.scores[0].cost_by_model[STRONG], 200_000)

    def test_all_failed_shows_no_cost(self):
        rows = [row(input_tokens=0, status=CONTEXT_OVERFLOW, score=0.0)]
        rep = build_report(rows, strong=STRONG, weak=WEAK)
        self.assertNotIn(STRONG, rep.scores[0].cost_by_model)
        self.assertIn("| -! |", to_markdown(rep))

    def test_negative_spread_does_not_outrank_zero(self):
        # B17: same strong accuracy and cost; "b-noisy" (weak beat strong,
        # spread -0.25) used to rank ahead of "a-robust" (spread 0.00).
        rows = []
        for i in range(4):
            rows.append(row(conversion="b-noisy", question_id=f"q{i}", score=0.0 if i == 2 else 1.0))
            rows.append(row(conversion="b-noisy", question_id=f"q{i}", model=WEAK, score=1.0))
            rows.append(row(conversion="a-robust", question_id=f"q{i}", score=0.0 if i == 2 else 1.0))
            rows.append(row(conversion="a-robust", question_id=f"q{i}", model=WEAK,
                            score=0.0 if i == 2 else 1.0))
        rep = build_report(rows, strong=STRONG, weak=WEAK)
        spreads = {cs.conversion: cs.spread for cs in rep.scores}
        self.assertAlmostEqual(spreads["b-noisy"], -0.25)
        self.assertAlmostEqual(spreads["a-robust"], 0.0)
        self.assertEqual([cs.conversion for cs in rep.scores], ["a-robust", "b-noisy"])

    def test_weak_only_footnote_does_not_mention_the_strong_tier(self):
        # B18
        md = to_markdown(build_report([row(model=WEAK)], strong=STRONG, weak=WEAK))
        self.assertNotIn("Spread =", md)
        self.assertNotIn(STRONG, md)
        self.assertIn(f"Cost is mean {WEAK} input tokens", md)

    def test_floor_footnote_uses_the_report_floor(self):
        rows = [row(score=0.4), row(model=WEAK, score=0.4)]
        md = to_markdown(build_report(rows, strong=STRONG, weak=WEAK, floor=0.5))
        self.assertIn("below the 50% accuracy floor", md)


class TestPricing(unittest.TestCase):
    """B21 / B13 / B15."""

    def test_current_list_prices(self):
        md = cost_summary([row(model="claude-sonnet-5", input_tokens=1_000_000,
                               output_tokens=0, max_tokens=512)])
        self.assertIn("$2.00", md)

    def test_dated_snapshot_prices_as_its_alias(self):
        md = cost_summary([row(model="claude-haiku-4-5-20251001", input_tokens=1_000_000,
                               output_tokens=0)])
        self.assertIn("$1.00", md)
        self.assertNotIn("No price on file", md)

    def test_model_specific_cache_read_rate(self):
        r = row(model="claude-fable-5-1", input_tokens=1000, cache_read_tokens=1000)
        self.assertAlmostEqual(billed_input_tokens(r), 25.0)
        self.assertAlmostEqual(billed_input_tokens(replace(r, model=STRONG)), 100.0)

    def test_uncached_current_row_is_not_called_uninstrumented(self):
        md = cost_summary([row(input_tokens=500, max_tokens=512)])
        self.assertNotIn("predate cache-component recording", md)
        md = cost_summary([row(input_tokens=500, max_tokens=0)])
        self.assertIn("predate cache-component recording", md)

    def test_judge_calls_are_costed(self):
        md = cost_summary([row(judge_model="claude-sonnet-4-6", judge_input_tokens=1_000_000,
                               judge_output_tokens=0, input_tokens=0, output_tokens=0)])
        self.assertIn("claude-sonnet-4-6 (judge)", md)
        self.assertIn("$3.00", md)


class TestSnapshotsAndServedModel(unittest.TestCase):
    """B15: dated snapshot IDs were refused and the served model not kept."""

    def test_snapshot_classifies_as_alias(self):
        self.assertEqual(base_model("claude-haiku-4-5-20251001"), "claude-haiku-4-5")
        self.assertIs(default_thinks("claude-haiku-4-5-20251001"), False)
        self.assertIsNone(default_thinks("claude-opus-9-20260101"))

    def test_cli_accepts_a_snapshot_id(self):
        # Reaching the missing corpus proves the model check passed.
        with self.assertRaises(FileNotFoundError):
            main(["run", "--corpus", "no-such-dir", "--out", "unused.md",
                  "--weak", "claude-haiku-4-5-20251001"])

    def test_newer_models_are_classified_as_thinking(self):
        # O8: these think whatever is sent, so they must be refused, not unknown.
        for m in ("claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1"):
            self.assertIs(default_thinks(m), True, m)

    def test_served_model_is_recorded_and_mixing_is_flagged(self):
        rows = [row(served_model="claude-haiku-4-5-20251001", model=WEAK),
                row(served_model="claude-haiku-4-5-20260301", model=WEAK, question_id="q2")]
        self.assertIn("more than one model version", regime_note(rows))
        self.assertNotIn("more than one model version", regime_note(rows[:1]))


def _stub_response(**kw):
    usage = SimpleNamespace(input_tokens=10, output_tokens=3,
                            cache_read_input_tokens=90, cache_creation_input_tokens=0)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="42")], usage=usage,
        stop_reason="end_turn", model="claude-haiku-4-5-20251001", **kw)


class _StubMessages:
    def __init__(self):
        self.created, self.counted = [], []

    def create(self, **kw):
        self.created.append(kw)
        return _stub_response()

    def count_tokens(self, **kw):
        self.counted.append(kw)
        return SimpleNamespace(input_tokens=7)


class TestAnthropicClientAgainstStubResponses(unittest.TestCase):
    """S3's missing coverage: answer()/count_input_tokens() against SDK-shaped
    objects, plus O2's PDF memo and B24's empty system prompt."""

    def setUp(self):
        self.messages = _StubMessages()
        self.client = AnthropicModelClient(client=SimpleNamespace(messages=self.messages))

    def test_answer_reads_usage_and_served_model(self):
        res = self.client.answer(model=WEAK, system="s", prompt="p",
                                 document=("text", "doc"))
        self.assertEqual(res.text, "42")
        self.assertEqual(res.input_tokens, 100)
        self.assertEqual(res.cache_read_tokens, 90)
        self.assertEqual(res.served_model, "claude-haiku-4-5-20251001")
        sent = self.messages.created[0]
        self.assertNotIn("thinking", sent)
        self.assertNotIn("temperature", sent)

    def test_pdf_is_encoded_once_per_file_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "s.pdf"
            pdf.write_bytes(b"%PDF-1.4 one")
            with mock.patch("decant_eval.models.base64.standard_b64encode",
                            wraps=__import__("base64").standard_b64encode) as enc:
                for _ in range(3):
                    self.client.answer(model=WEAK, system="s", prompt="p",
                                       document=("pdf", pdf))
                self.assertEqual(enc.call_count, 1)
                pdf.write_bytes(b"%PDF-1.4 two, longer")
                blocks = self.client._document_blocks(("pdf", pdf))
                self.assertEqual(enc.call_count, 2)
            self.assertIn("cache_control", blocks[-1])
            # the memoized block is not mutated by the breakpoint
            self.assertNotIn("cache_control", self.client._pdf_block(pdf))

    def test_count_tokens_omits_an_empty_system_prompt(self):
        self.assertEqual(self.client.count_input_tokens(model=WEAK, system="", prompt="x"), 7)
        self.assertNotIn("system", self.messages.counted[0])
        self.client.count_input_tokens(model=WEAK, system="sys", prompt="x")
        self.assertEqual(self.messages.counted[1]["system"], "sys")


if __name__ == "__main__":
    unittest.main()
