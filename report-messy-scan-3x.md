# Decant eval report

| conversion | claude-haiku-4-5 | claude-opus-4-8 | cost (strong tok) | spread | n |
| --- | --- | --- | --- | --- | --- |
| decant | 0.53 | 0.80 | 168711 | +0.27 | 1 |
| raw | 0.00 | 0.80 | 212599 | +0.80 | 1 |
| unlimited-ocr | 0.60 | 0.73 | 138342 | +0.13 | 1 |
| docling | 0.60 | 0.70 | 114611 | +0.10 | 1 |
| decant-plain | 0.50 | 0.60 | 43644 | +0.10 | 1 |
| markitdown | 0.50 | 0.60 | 79673 | +0.10 | 1 |

_Spread = claude-opus-4-8 accuracy - claude-haiku-4-5 accuracy; lower means the conversion transfers meaning robustly to the weaker reader. Cost is claude-opus-4-8 input tokens (tiers tokenize differently)._

## By answer source

| case | source | conversion | claude-haiku-4-5 | claude-opus-4-8 | n |
| --- | --- | --- | --- | --- | --- |
| messy-scan | figure-2 | decant | 0.00 | 0.00 | 1 |
| messy-scan | figure-2 | decant-plain | 0.00 | 0.00 | 1 |
| messy-scan | figure-2 | docling | 0.00 | 0.00 | 1 |
| messy-scan | figure-2 | markitdown | 0.00 | 0.00 | 1 |
| messy-scan | figure-2 | raw | 0.00 | 0.00 | 1 |
| messy-scan | figure-2 | unlimited-ocr | 0.00 | 0.00 | 1 |
| messy-scan | figure-45 | decant | 0.00 | 1.00 | 1 |
| messy-scan | figure-45 | decant-plain | 0.00 | 0.00 | 1 |
| messy-scan | figure-45 | docling | 1.00 | 1.00 | 1 |
| messy-scan | figure-45 | markitdown | 0.00 | 0.00 | 1 |
| messy-scan | figure-45 | raw | 0.00 | 1.00 | 1 |
| messy-scan | figure-45 | unlimited-ocr | 1.00 | 1.00 | 1 |
| messy-scan | figure-6 | decant | 0.00 | 0.00 | 1 |
| messy-scan | figure-6 | decant-plain | 0.00 | 0.00 | 1 |
| messy-scan | figure-6 | docling | 0.00 | 0.00 | 1 |
| messy-scan | figure-6 | markitdown | 0.00 | 0.00 | 1 |
| messy-scan | figure-6 | raw | 0.00 | 0.00 | 1 |
| messy-scan | figure-6 | unlimited-ocr | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | decant | 0.00 | 1.00 | 1 |
| messy-scan | table-10 | decant-plain | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | docling | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | markitdown | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | raw | 0.00 | 1.00 | 1 |
| messy-scan | table-10 | unlimited-ocr | 0.00 | 0.33 | 1 |

_Accuracy sliced by where the answer lives in the source (the questions' `source` tags). Tags are case-scoped; conversions within one case+source group answered the same questions and compare directly._

## Cost

| model | billed input tok | output tok | USD (list) |
| --- | --- | --- | --- |
| claude-haiku-4-5 | 14,024,115 | 9,197 | $14.07 |
| claude-opus-4-8 | 22,727,454 | 5,853 | $113.78 |
| **total** | | | **$127.85** |

_Billed input counts cache reads at 0.1x and writes at 1.25x base rate, so it is well below the raw input-token total on a cached run. List prices only — no discounts, batch rates, or negotiated terms._
_WARNING: 330 row(s) predate cache-component recording and are priced as fully uncached. The real figure is lower; treat this as a ceiling._