**Slot: obscure or private document.**

The counterpart to `public-famous/` — a document the model almost certainly
hasn't seen (an internal doc, a niche/unpublished PDF, something recent
enough to postdate training). Every correct answer here has to have come
from the conversion, not memory, so this slot is where the reliability
spread and accuracy numbers are cleanest to trust at face value.

To fill this slot: drop `source.pdf` here, run MarkItDown/Docling/Decant
against it into `conversions/`, then author `questions.json` (see
`corpus/README.md` for the shape and the gold-from-source rule). This
directory is skipped by the harness until `questions.json` exists.
