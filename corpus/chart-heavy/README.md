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
