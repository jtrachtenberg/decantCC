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
    decant.pdf              # optional figures companion: a PDF with the same
                            # stem rides with that conversion — the model gets
                            # the text AND the figures PDF together (labels in
                            # the text point into the figures)
    markitdown.md
    docling.md              # each file = one conversion in the arena
```

A companion PDF must have a non-empty same-stem conversion to attach to —
an orphan (typo'd stem, or figures next to a still-empty `.md`) fails the
load loudly rather than silently dropping the figures from a billed run.
Companion pages are billed as images; `tokens.py` pre-run estimates don't
include them (the run's own `usage` does).

A case is only picked up by `load_corpus()` once it has **both** a
`questions.json` (or `.yaml`/`.yml`) and at least one **non-empty**
`.md`/`.txt` file in `conversions/` — until then it's just a placeholder
directory and is silently skipped, so it's safe to scaffold slots ahead of
having real content, in either order (questions-first or conversions-first).
Empty or whitespace-only conversion files are ignored, so a not-yet-written
`decant.md` won't be scored as a blank (uniformly wrong) arena entry.

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
3. `questions.json` — questions spanning the graded types
   (`numeric` / `exact` / `set` / `ordered_list` / `open`), gold answers
   from the source. Each question may carry an optional `source` tag naming
   where in the document its answer lives (`"figure-12"`, `"table-3"`,
   `"text"` — free-form, case-scoped); the report then slices accuracy by
   tag, so figure-tagged questions reveal what a figures-companion PDF
   contributes and which figures earn their image tokens.

## Diversity checklist for corpus selection

Six placeholder slots below cover the axes worth stressing. Aim for 5–8
cases total, not volume — each slot should be filled with one real document,
not many:

- **`clean-text/`** — clean, text-heavy native PDF. The easy baseline; every
  conversion should score well here, so a bad score is a real bug signal.
- **`table-heavy/`** — dense tabular data. Stresses row/column binding
  (the "confidently wrong table" failure mode the `sample-invoice` fixture
  simulates synthetically).
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

The harness's synthetic end-to-end fixture (a fake invoice with a clean and
a deliberately garbled conversion) lives at `tests/fixtures/sample-invoice/`,
**not** in this directory: its `clean`/`garbled` arms share no case with the
real conversion arms, so inside the corpus it would leave the report with no
common case at all — nothing comparable. Keep test-only cases out of
`corpus/`.
