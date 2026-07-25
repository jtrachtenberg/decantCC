# table-heavy — non-arena conversions

Files here are **not scored**: `load_corpus` only reads `conversions/`. They
are kept as artifacts of experiments that ran against this case.

- `unlimited-ocr.md` — Baidu Unlimited-OCR output, moved out of the arena on
  2026-07-25. This document has **no broken-font or image-only pages**
  (`decant.md` contains zero "could not be decoded" markers), so OCR cannot be
  the sole carrier of any answer here; scoring it only made the OCR arm span 2
  of 6 cases and skewed the aggregate ranking against arms measured on all 6.
  The OCR arm is now scoped to `messy-scan`, the case that actually has 32
  broken-font and 20 image-only pages. See `PLAN.md`.
