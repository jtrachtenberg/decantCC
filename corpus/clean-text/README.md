**Slot: clean, text-heavy native PDF.**

The easy baseline. Prose-dominant, no meaningful tables or charts, exported
from a real document (not scanned). Every conversion in the arena should
score close to ceiling here — if one doesn't, that's a real bug, not corpus
noise. Good candidate: a report, article, or documentation PDF with minimal
layout complexity.

To fill this slot: drop `source.pdf` here, run MarkItDown/Docling/Decant
against it into `conversions/`, then author `questions.json` (see
`corpus/README.md` for the shape and the gold-from-source rule). This
directory is skipped by the harness until `questions.json` exists.
