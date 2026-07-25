# clean-text-image-only — derived recognition-tier A/B (RETIRED 2026-07-25)

> **Retired.** The case was removed from `corpus/` on 2026-07-25: it was built
> to exercise the unlimited-ocr arm, and its `questions.json` was a verbatim
> copy of `clean-text/`'s, so it added duplicate question ids and a case with
> no Decant arm to the arena without adding independent signal about the codec.
> This file and `report-clean-text-image-only.md` are the surviving record; the
> case is reproducible from `corpus/clean-text/source.pdf` by the recipe below.
> Its two durable findings are carried into `PLAN.md`: Decant declines
> image-only PDFs outright, and OCR text beat raw vision on the weak reader at
> less than half the tokens.

**Derived case**, not an independent document: `source.pdf` is
`corpus/clean-text/source.pdf` with every page re-rendered as a 200-dpi
JPEG (quality 80, PyMuPDF + Pillow, 2026-07-23) and rebuilt into a PDF with
**no text layer** — verified at build time: `get_text()` across all 42 pages
extracts **0 characters**.

This is the controlled A/B suggested in `corpus/README.md` ("Known gap"):
`questions.json` is copied **verbatim** from `clean-text/`, so the gold
answers are identical and still satisfy the hard rule (they were authored
from the same source document, human-verified). What changes is only the
representation available to converters:

- **parsers** (Decant/pdf.js, MarkItDown, plain extraction) see no text and
  should pass through ~nothing — a floor, not a bug;
- **recognition** (OCR conversions, and the raw-upload arm's vision) must
  actually read the pixels.

Scores here vs. the native `clean-text/` case isolate the recognition tier
with everything else held constant: same content, same questions, same gold.

Conversions present:

- `unlimited-ocr.md` — Baidu Unlimited-OCR (Transformers path, gundam mode,
  generation capped at 8192 tokens/page; degenerate pages, if any, are
  flagged inline with `[OCR unreliable — page N]`).
- `raw` is automatic (the image-only `source.pdf` fed as a document block).

## Parser-floor results (measured 2026-07-23, tool defaults)

The parser arms were run against this `source.pdf`; the floor turned out to
be *no output*, not low scores:

- **MarkItDown 0.1.6** — exit 0, **0-byte output** (pdfminer finds no text
  layer). The empty `conversions/markitdown.md` is kept as the artifact; the
  harness ignores empty conversions by design, so it does not appear as a
  scored arm.
- **Decant 0.3.1** — `--mode markdown` and `--mode figures` both exit 10:
  **`passthrough (no-text)`**. Decant's classifier detects the missing text
  layer and declines to convert; its product answer is "send the original",
  so decant's effective arm here *is* the `raw` upload arm. (This is the
  exact hook where a Force-OCR / auto-escalation tier would plug in.)
- **Docling 2.111.0** — defaults include built-in OCR, so it produces a real
  conversion (`docling.md`) and is scored as a recognition-tier arm, not a
  parser floor.

Never copy conversions from `clean-text/` into this case; those were
produced from the native text layer this case exists to remove.

## Eval snapshot (2026-07-23, `report-clean-text-image-only.md`)

| arm | Haiku 4.5 | Opus 4.8 | strong tokens | spread |
|---|---|---|---|---|
| unlimited-ocr | 1.00 | 1.00 | 29,466 | 0.00 |
| raw (vision) | 0.88 | 1.00 | 65,381 | +0.12 |
| docling (OCR) | 0.00* | 1.00 | 244,317 | +1.00 |

\* Not a comprehension failure: docling's ~244K-token conversion **exceeds
Haiku's 200K context window**, so every weak-reader call failed outright
(`input_tokens=0` in the audit trail). The 8× token bloat *is* the failure —
the representation stops fitting smaller readers. Control arm: clean (no
question answerable without a document).
