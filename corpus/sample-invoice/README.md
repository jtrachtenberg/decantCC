Sample eval case (also the harness's end-to-end test fixture).

`clean.md` is a faithful conversion; `garbled.md` interleaves the table so row
bindings are lost — the "confidently wrong table" failure mode. A good harness
shows `clean` scoring well on both model tiers and `garbled` scoring worse,
especially for the weaker model (a larger reliability spread).

Gold answers were written from the (fictional) source, not from either
conversion — the hard rule. There is no real `source.pdf` here; production
cases include one for reference.
