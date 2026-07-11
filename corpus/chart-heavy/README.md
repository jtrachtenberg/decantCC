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

- `conversions/decant.md` + `conversions/decant.pdf` — Decant in
  **md+charts** mode: markdown plus its extracted-figures companion PDF
  (same stem = paired; the PDF's labels tie each figure back to its place
  in the markdown). The harness feeds both to the model together.
- `conversions/decant-plain.md` — Decant in **markdown-only** mode. Not a
  byte copy of decant.md: the md+charts markdown carries a few extra
  figure-location labels that the md-only output lacks.

Scoring both compares Decant's two product configurations — the choice a
Decant user actually faces. Any accuracy or reliability-spread gap between
the arms is attributable to choosing the charts option (figures companion
plus its location labels), including its token cost, since companion pages
are billed as images.

Both modes must be generated from the **same Decant build in the same
sitting** — a bug fix landing between the two runs makes the ablation
compare two different converters.
Worth watching against `markitdown`/`docling`, which already diverge here —
MarkItDown keeps the "CERN in Figures" pie-chart numbers (they're text in
the source's layout layer) while Docling drops them as graphic regions.
