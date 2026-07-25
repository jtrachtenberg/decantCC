# Decant eval report

| conversion | claude-haiku-4-5 | claude-opus-4-8 | cost (strong tok) | spread | n |
| --- | --- | --- | --- | --- | --- |
| docling | 0.60 | 0.70 | 114611 | +0.10 | 1 |
| decant | 0.50 | 0.70 | 116795 | +0.20 | 1 |
| unlimited-ocr | 0.60 | 0.70 | 138342 | +0.10 | 1 |
| raw | 0.00 | 0.70 | 212599 | +0.70 | 1 |
| decant-plain | 0.50 | 0.60 | 60976 | +0.10 | 1 |
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
| messy-scan | table-10 | decant | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | decant-plain | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | docling | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | markitdown | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | raw | 0.00 | 0.00 | 1 |
| messy-scan | table-10 | unlimited-ocr | 0.00 | 0.00 | 1 |

_Accuracy sliced by where the answer lives in the source (the questions' `source` tags). Tags are case-scoped; conversions within one case+source group answered the same questions and compare directly._

## Memory-contamination control

_No question was answered correctly with no document — the arena scores reflect the representation, not the model's prior knowledge._