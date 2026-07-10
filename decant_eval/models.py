"""Model client abstraction.

The runner talks to this interface, never to the Anthropic SDK directly, so the
whole harness unit-tests offline with FakeModelClient (no API key, no network).
AnthropicModelClient is the real thing.

A `document` may accompany the question:

    None                       no document — the memory-contamination control arm
    ("text", markdown)         a cached conversion (a .md/.txt file being scored)
    ("text+pdf", (md, Path))   a conversion plus its figures-companion PDF
                               (e.g. decant.md + decant.pdf): the text carries
                               labels that reference figures the companion holds
    ("pdf", Path)              the source PDF itself — the *raw upload* arena
                               anchor, the baseline the thesis is measured against

The document is sent as its own content block(s) with a single `cache_control`
breakpoint on the last of them — prefix caching then covers the whole document
group — so the loop re-uses the cached document across every question about it
instead of re-billing the full input each time (~90% cheaper on large
documents). The question follows in an uncached block. All document kinds share
this path so caching applies uniformly. Note a companion PDF is billed as
rendered page images on every (cache-miss) request — that token weight is part
of what the eval measures for such arms.

Deliberate: no thinking / effort is requested. The harness measures whether a
*representation transfers meaning* — thinking off keeps a strong model from
reasoning its way around a corrupted conversion, which would confound the
signal, and it sharpens the strong-vs-weak reliability spread. (It's a knob we
can revisit; see README.) Note Haiku 4.5 rejects the `effort` param outright,
so we send neither.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class AnswerResult:
    text: str
    input_tokens: int
    output_tokens: int


class ModelClient(Protocol):
    def answer(
        self, *, model: str, system: str, prompt: str, max_tokens: int = 512, document=None
    ) -> AnswerResult: ...


def _render(document) -> str:
    """The document as text for the fake client / accounting. `("pdf", path)`
    has no text, so its bytes stand in for length."""
    if document is None:
        return ""
    kind, payload = document
    if kind == "text":
        return f"DOCUMENT:\n{payload}\n\n"
    if kind == "text+pdf":
        text, path = payload
        return f"DOCUMENT:\n{text}\n\n[PDF {Path(path).name}]\n\n"
    return f"[PDF {Path(payload).name}]\n\n"  # raw upload — no extractable text


class AnthropicModelClient:
    """Real client over the Anthropic SDK. `client` defaults to a zero-arg
    Anthropic() (resolves ANTHROPIC_API_KEY or an `ant auth login` profile)."""

    def __init__(self, client=None):
        if client is None:
            from anthropic import Anthropic  # imported lazily so tests need no SDK

            client = Anthropic()
        self._client = client

    @staticmethod
    def _pdf_block(path):
        data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
        }

    def _document_blocks(self, document) -> list:
        """The document as content blocks ([] when there is no document). Text
        precedes its figures companion so the text's labels read as references
        to the figures that follow."""
        if document is None:
            return []
        kind, payload = document
        if kind == "text":
            blocks = [{"type": "text", "text": f"DOCUMENT:\n{payload}"}]
        elif kind == "text+pdf":
            text, path = payload
            blocks = [{"type": "text", "text": f"DOCUMENT:\n{text}"}, self._pdf_block(path)]
        elif kind == "pdf":
            blocks = [self._pdf_block(payload)]
        else:  # pragma: no cover - guarded by the runner
            raise ValueError(f"unknown document kind {kind!r}")
        # One breakpoint on the last block caches the whole document prefix;
        # the question after it is not cached.
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    def answer(self, *, model, system, prompt, max_tokens=512, document=None) -> AnswerResult:
        # No thinking/effort (see module docstring): omit thinking → Opus 4.8
        # runs without it; Haiku 4.5 would 400 on effort, so it's absent too.
        content = list(self._document_blocks(document))
        content.append({"type": "text", "text": prompt})
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        # cache reads/writes still count as input the run paid for; usage.input_tokens
        # is the uncached remainder, so add the cached portions back for a true cost.
        usage = resp.usage
        input_tokens = (
            usage.input_tokens
            + getattr(usage, "cache_read_input_tokens", 0)
            + getattr(usage, "cache_creation_input_tokens", 0)
        )
        return AnswerResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
        )

    def count_input_tokens(self, *, model, system, prompt) -> int:
        """A-priori input-token cost of feeding (system + prompt) to `model` —
        the count_tokens endpoint, model-specific. Used to estimate a
        conversion's token weight before a run (see tokens.py)."""
        resp = self._client.messages.count_tokens(
            model=model,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.input_tokens


class FakeModelClient:
    """Scripted client for tests. `responder(model, system, prompt) -> str`
    supplies the answer; the document (if any) is rendered into `prompt` so the
    responder sees exactly what the real model would. Token counts approximate
    at ~4 chars/token so the accounting paths still exercise real numbers."""

    def __init__(self, responder):
        self._responder = responder
        self.calls: list[tuple[str, str]] = []  # (model, rendered prompt) for assertions

    def answer(self, *, model, system, prompt, max_tokens=512, document=None) -> AnswerResult:
        full = _render(document) + prompt
        self.calls.append((model, full))
        text = self._responder(model, system, full)
        return AnswerResult(
            text=text,
            input_tokens=max(1, (len(system) + len(full)) // 4),
            output_tokens=max(1, len(text) // 4),
        )
