**Slot: dense tabular data.**

A document where most of the signal lives in tables — financial statements,
data appendices, comparison tables. Stresses row/column binding: the
"confidently wrong table" failure mode where a conversion interleaves rows
and produces a plausible-looking but factually scrambled table (the class of
bug `sample-invoice/`'s `garbled.md` simulates synthetically — this slot is
the real-document version). Numeric and set-type questions pair well here.

To fill this slot: drop `source.pdf` here, run MarkItDown/Docling/Decant
against it into `conversions/`, then author `questions.json` (see
`corpus/README.md` for the shape and the gold-from-source rule). This
directory is skipped by the harness until `questions.json` exists.
