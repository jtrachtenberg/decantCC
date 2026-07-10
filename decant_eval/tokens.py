"""A-priori conversion token cost.

During a run the exact per-answer cost comes from each response's `usage`
(runner.py), so this module is only for estimating a conversion's token weight
*before* running — e.g. to rank conversions by size, or budget a run. It wraps
the model-specific count_tokens endpoint, with a portable chars/4 fallback so a
rough number is available with no client.

Caveat: these estimates cover a conversion's *text* only. A figures-companion
PDF (e.g. decant.pdf riding with decant.md) is billed as rendered page images,
which this module does not estimate — the run's own `usage` is the source of
truth for such arms.
"""

from __future__ import annotations

CHARS_PER_TOKEN = 4  # portable rough figure; never a substitute for count_tokens


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def conversion_cost(text: str, *, client=None, model: str = "claude-opus-4-8") -> int:
    """Input tokens for a conversion. Uses the count_tokens endpoint when an
    AnthropicModelClient is given (accurate, model-specific); otherwise the
    chars/4 estimate."""
    if client is not None and hasattr(client, "count_input_tokens"):
        return client.count_input_tokens(model=model, system="", prompt=text)
    return estimate_tokens(text)
