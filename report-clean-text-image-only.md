# Decant eval report

| conversion | claude-haiku-4-5 | claude-opus-4-8 | cost (strong tok) | spread | n |
| --- | --- | --- | --- | --- | --- |
| unlimited-ocr | 1.00 | 1.00 | 29466 | +0.00 | 1 |
| raw | 0.88 | 1.00 | 65381 | +0.12 | 1 |
| docling | 0.00 | 1.00 | 244317 | +1.00 | 1 |

_Spread = claude-opus-4-8 accuracy - claude-haiku-4-5 accuracy; lower means the conversion transfers meaning robustly to the weaker reader. Cost is claude-opus-4-8 input tokens (tiers tokenize differently)._

## By answer source

| case | source | conversion | claude-haiku-4-5 | claude-opus-4-8 | n |
| --- | --- | --- | --- | --- | --- |
| clean-text-image-only | figure-34 | docling | 0.00 | 1.00 | 1 |
| clean-text-image-only | figure-34 | raw | 1.00 | 1.00 | 1 |
| clean-text-image-only | figure-34 | unlimited-ocr | 1.00 | 1.00 | 1 |
| clean-text-image-only | text | docling | 0.00 | 1.00 | 1 |
| clean-text-image-only | text | raw | 1.00 | 1.00 | 1 |
| clean-text-image-only | text | unlimited-ocr | 1.00 | 1.00 | 1 |

_Accuracy sliced by where the answer lives in the source (the questions' `source` tags). Tags are case-scoped; conversions within one case+source group answered the same questions and compare directly._

## Memory-contamination control

_No question was answered correctly with no document — the arena scores reflect the representation, not the model's prior knowledge._