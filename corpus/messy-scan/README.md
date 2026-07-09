**Slot: scanned or structurally messy.**

A document that stresses layout recovery — scanned pages, multi-column
text, rotated content, or dense mixed layouts. This is the real-world class
of bug that produces garbled conversions in production (the WHO-report
column-interleave case lived here). The point isn't to find the single
hardest document available — a moderately messy real-world PDF is more
representative than a pathological worst case.

To fill this slot: drop `source.pdf` here, run MarkItDown/Docling/Decant
against it into `conversions/`, then author `questions.json` (see
`corpus/README.md` for the shape and the gold-from-source rule). This
directory is skipped by the harness until `questions.json` exists.
