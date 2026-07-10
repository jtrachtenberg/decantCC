**Slot: figures and charts.**

A document where bar/line/pie charts or other figures carry real
information (not just decoration). This is where Decant's chart-fidelity
tiers are meant to differentiate it from MarkItDown/Docling, so it's a good
place to see conversions diverge sharply rather than cluster together.
Favor a document with a few distinct chart types over one with many
near-identical charts.

To fill this slot: drop `source.pdf` here, run MarkItDown/Docling/Decant
against it into `conversions/`, then author `questions.json` (see
`corpus/README.md` for the shape and the gold-from-source rule). This
directory is skipped by the harness until `questions.json` exists.

**Decant ablation for this case.** Because this is where chart fidelity is
supposed to matter, the Decant arm is split into two arena entries:

- `conversions/decant.md` + `conversions/decant.pdf` — Decant's markdown
  with its extracted-figures companion PDF (same stem = paired; the PDF's
  labels tie each figure back to its place in the markdown). The harness
  feeds both to the model together — this is the product being tested.
- `conversions/decant-plain.md` — the *same* Decant markdown with **no**
  companion PDF (text only).

Scoring both isolates the marginal value of the chart tiers: `decant` vs
`decant-plain` differ only by the figures companion, so any accuracy or
reliability-spread gap between them is attributable to it and nothing else —
including its token cost, since companion pages are billed as images.
Worth watching against `markitdown`/`docling`, which already diverge here —
MarkItDown keeps the "CERN in Figures" pie-chart numbers (they're text in
the source's layout layer) while Docling drops them as graphic regions.
