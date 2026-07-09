**Slot: well-known public document.**

A document likely to appear in a model's training data (e.g. a well-known
report or publication — the WHO report already flagged elsewhere as
"certainly memorized" is a good candidate). This slot exists specifically to
exercise the memory-contamination control arm (`run_control` — every
question asked with no document present). The model may answer correctly
from memory alone, and that has to show up in the report as a flagged
question, not get mistaken for the conversion successfully transferring
meaning.

To fill this slot: drop `source.pdf` here, run MarkItDown/Docling/Decant
against it into `conversions/`, then author `questions.json` (see
`corpus/README.md` for the shape and the gold-from-source rule). This
directory is skipped by the harness until `questions.json` exists.
