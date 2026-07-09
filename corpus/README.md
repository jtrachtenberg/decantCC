# Corpus

Each subdirectory is one **case** — a source document plus the questions
authored against it and the candidate conversions to score.

```
corpus/<case>/
  source.pdf              # the original — reference only, never fed as text;
                           # if present it also becomes the implicit "raw"
                           # arena entry (fed as a document block, cached)
  questions.json           # or questions.yaml (needs pyyaml)
  conversions/
    decant.md
    markitdown.md
    docling.md              # each file = one conversion in the arena
```

A case is only picked up by `load_corpus()` once it has a `questions.json`
(or `.yaml`/`.yml`) — until then it's just a placeholder directory and is
silently skipped, so it's safe to scaffold slots ahead of having real content.

## The hard rule

Gold answers in `questions.json` come from the **source document**, drafted
and verified by a human — never from a conversion. Grading a conversion
against an answer pulled from that same conversion tests conformance to the
converter, not correctness.

## Per-case checklist

1. `source.pdf` — the real document.
2. `conversions/*.md` — run MarkItDown, Docling, and Decant's own output
   against it and drop each in. `raw` is automatic (see above) — don't
   hand-author it unless you want to override the auto-generated one.
3. `questions.json` — 3–6 questions spanning the four graded types
   (`numeric` / `exact` / `set` / `open`), gold answers from the source.

## Diversity checklist for corpus selection

Six placeholder slots below cover the axes worth stressing. Aim for 5–8
cases total, not volume — each slot should be filled with one real document,
not many:

- **`clean-text/`** — clean, text-heavy native PDF. The easy baseline; every
  conversion should score well here, so a bad score is a real bug signal.
- **`table-heavy/`** — dense tabular data. Stresses row/column binding
  (the "confidently wrong table" failure mode `sample-invoice/` simulates).
- **`chart-heavy/`** — figures and charts. Decant's differentiator
  (chart-fidelity tiers); a good place to see conversions diverge sharply.
- **`messy-scan/`** — scanned, multi-column, or otherwise structurally messy.
  Real-world garbled-conversion risk (the WHO-report column-interleave class
  of bug lived here).
- **`public-famous/`** — a well-known public document. Exercises the
  memory-contamination control arm (`run_control` — no-document questions);
  the model may answer from training data, and that needs to be visible in
  the report, not mistaken for the conversion transferring meaning.
- **`private-novel/`** — an obscure/private document the model has never
  seen. The clean-signal counterpart to `public-famous/` — no memory
  confound, so scores here isolate representation transfer.

`sample-invoice/` is the harness's synthetic end-to-end test fixture (clean
vs. deliberately garbled table) — it's not part of the real corpus and stays
put regardless of what lands in the slots above.
