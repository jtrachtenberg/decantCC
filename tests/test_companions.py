"""Figures-companion PDFs: a <stem>.pdf in conversions/ rides with the
same-stem .md/.txt conversion and is fed to the model alongside its text
(the decant.md + decant.pdf convention). Offline."""

import json
import tempfile
import unittest
from pathlib import Path

from decant_eval.corpus import load_case
from decant_eval.models import AnthropicModelClient, FakeModelClient
from decant_eval.runner import RAW, _arena_entries, run_case

STRONG = "claude-opus-4-8"

FAKE_PDF = b"%PDF-1.4 fake figures"


def make_case(tmp, *, companion=True, orphan=False, empty_md_with_pdf=False):
    d = Path(tmp) / "c1"
    conv = d / "conversions"
    conv.mkdir(parents=True)
    (d / "questions.json").write_text(
        json.dumps({"questions": [
            {"id": "total", "question": "total?", "gold": "1250.00", "type": "numeric", "tolerance": 0.01},
        ]}), encoding="utf-8",
    )
    (conv / "markitdown.md").write_text("Total: 1250.00 USD", encoding="utf-8")
    (d / "source.pdf").write_bytes(b"%PDF-1.4 source")
    if companion:
        (conv / "decant.md").write_text("Total: 1250.00 USD [see FIGURE-1]", encoding="utf-8")
        (conv / "decant.pdf").write_bytes(FAKE_PDF)
    if orphan:
        (conv / "nosuch.pdf").write_bytes(FAKE_PDF)
    if empty_md_with_pdf:
        (conv / "half.md").write_text("   \n", encoding="utf-8")
        (conv / "half.pdf").write_bytes(FAKE_PDF)
    return d


class TestCompanionDiscovery(unittest.TestCase):
    def test_same_stem_pdf_is_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = load_case(make_case(tmp))
            self.assertIn("decant", case.companions)
            self.assertEqual(case.companions["decant"].name, "decant.pdf")
            self.assertNotIn("markitdown", case.companions)  # no pdf, no pairing

    def test_orphan_companion_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                load_case(make_case(tmp, orphan=True))
            self.assertIn("nosuch", str(ctx.exception))

    def test_companion_of_empty_conversion_is_loud(self):
        # An empty half.md is skipped by the empty-file guard, so half.pdf has
        # nothing to attach to: the arm is half-written and must not run silently.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as ctx:
                load_case(make_case(tmp, empty_md_with_pdf=True))
            self.assertIn("half", str(ctx.exception))


class TestCompanionArena(unittest.TestCase):
    def test_arena_entry_carries_text_and_companion(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = load_case(make_case(tmp))
            entries = dict(_arena_entries(case, raw_arena=True))
            kind, payload = entries["decant"]
            self.assertEqual(kind, "text+pdf")
            text, path = payload
            self.assertIn("FIGURE-1", text)
            self.assertEqual(Path(path).name, "decant.pdf")
            self.assertEqual(entries["markitdown"][0], "text")  # untouched
            self.assertEqual(entries[RAW], ("pdf", case.source))  # baseline untouched

    def test_fake_client_sees_text_and_pdf_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = load_case(make_case(tmp))
            client = FakeModelClient(lambda m, s, p: "1250.00")
            run_case(case, client=client, models=[STRONG], raw_arena=False)
            decant_prompts = [p for _, p in client.calls if "FIGURE-1" in p]
            self.assertTrue(decant_prompts)
            self.assertIn("[PDF decant.pdf]", decant_prompts[0])
            self.assertIn("DOCUMENT:", decant_prompts[0])


class TestCompanionBlocks(unittest.TestCase):
    def _blocks(self, document):
        # _document_blocks never touches the SDK client, so a dummy suffices.
        return AnthropicModelClient(client=object())._document_blocks(document)

    def test_text_plus_pdf_is_two_blocks_one_breakpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "decant.pdf"
            pdf.write_bytes(FAKE_PDF)
            blocks = self._blocks(("text+pdf", ("labelled text", pdf)))
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0]["type"], "text")
            self.assertIn("labelled text", blocks[0]["text"])
            self.assertEqual(blocks[1]["type"], "document")
            self.assertEqual(blocks[1]["source"]["media_type"], "application/pdf")
            # one breakpoint, on the last block: prefix caching covers both
            self.assertNotIn("cache_control", blocks[0])
            self.assertEqual(blocks[1]["cache_control"], {"type": "ephemeral"})

    def test_plain_kinds_unchanged(self):
        blocks = self._blocks(("text", "just text"))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(self._blocks(None), [])


if __name__ == "__main__":
    unittest.main()
