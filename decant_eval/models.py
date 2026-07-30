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

No thinking and no sampling parameters are sent. The harness measures whether a
*representation transfers meaning* — thinking off keeps a strong model from
reasoning its way around a corrupted conversion, which would confound the
signal, and it sharpens the strong-vs-weak reliability spread.

CAREFUL: "no thinking" is a property of the request *as this model interprets
it*, not of the request itself. Omitting `thinking` means thinking-off only on
models whose default is off — Opus 4.8/4.7/4.6, Sonnet 4.6/4.5, Haiku 4.5. On
Opus 5, Sonnet 5, and Fable 5 the same omitted parameter runs *adaptive
thinking*, so pointing --strong at one of those silently inverts the property
the paragraph above rests on, with no code change and no error. Two
consequences beyond the confound itself: thinking bills as output tokens with
no separate usage field (so the split is unrecoverable from the response), and
it is charged against `max_tokens` alongside the visible answer, so the 512
default below stops being an answer budget. THINKING_OFF_BY_DEFAULT encodes
the split and cli.py refuses an unvetted model on that basis; keep it current
rather than relying on this comment.

Neither knob is a simple on/off across the two tiers, which is why a thinking
arm was dropped rather than built (2026-07-24):

  - `budget_tokens` is REMOVED on Opus 4.7+ (400). Opus 4.8 takes
    `thinking={"type": "adaptive"}` + `output_config={"effort": ...}`; Haiku 4.5
    still takes `budget_tokens` and rejects `effort` outright. "Thinking budget"
    is not the same knob on the two tiers, so a thinking arm would weaken the
    tier comparability the reliability spread depends on.
  - `temperature`/`top_p`/`top_k` are likewise REMOVED on Opus 4.7+ (400), so
    sampling cannot be pinned on the strong tier. Runs therefore sample at the
    API default (temperature 1.0) and per-cell variance is real — see
    `repeats` in runner.py, which is the only variance control available.

Truncation: a response can end because the model finished ("end_turn") or
because it ran out of budget ("max_tokens") — a partial answer that grades as
wrong. `AnswerResult.stop_reason` carries that distinction to the runner so the
audit trail shows which happened. It has never fired at the 512-token default
(the largest answer across every billed trail on disk is 363 tokens, in
messy-scan-3x — 71% of the cap, so the headroom is thinner than it looks and a
thinking model would have none), so this is
a guard rather than a fix for an observed problem — but a silently truncated
answer grades identically to a wrong one, and the report would then read a
budget artifact as a representation failure.

answer() raises on API failure rather than returning a sentinel result. The
runner classifies the exception (see runner._failure_status): a deterministic
failure — the document exceeds the model's context window — becomes a
status-tagged score-0 row, while transient errors propagate and crash the run
so --resume retries them instead of freezing a 0 into the audit trail.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# What omitting the `thinking` parameter means, per model. answer() never sends
# it, so this table -- not the request -- is what decides whether a run thinks.
# It is the arena's central assumption written down: every published number was
# measured with the strong tier NOT thinking, and a model in the second set
# breaks that silently rather than loudly (see the module docstring).
#
# Verified against the Anthropic model docs 2026-07-30. A model missing from
# both sets is UNKNOWN, not assumed safe -- cli.py refuses to run one until it
# is classified here, which is the point: the failure should be a startup error
# for a human to resolve, not a quietly different measurement regime.
THINKING_OFF_BY_DEFAULT = frozenset({
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5",
    "claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-4-5",
})
# Omitting `thinking` runs ADAPTIVE thinking on these -- thinking tokens bill as
# output, are charged against max_tokens, and vary with the input, so the two
# arms think different amounts by construction.
THINKING_ON_BY_DEFAULT = frozenset({
    "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-mythos-5",
})


def default_thinks(model: str):
    """True if `model` thinks when `thinking` is omitted, False if it does not,
    None if the model is unclassified. None is deliberately distinct from False:
    an unrecognized model is an open question, not a safe default."""
    if model in THINKING_OFF_BY_DEFAULT:
        return False
    if model in THINKING_ON_BY_DEFAULT:
        return True
    return None


@dataclass(frozen=True)
class AnswerResult:
    text: str
    # Total input the request covered: uncached + cache reads + cache writes.
    # An upper bound on what was billed, never the billed figure itself — the
    # three tiers price differently (see cache_read_tokens).
    input_tokens: int
    output_tokens: int
    # The cached portions of input_tokens, kept separate so actual spend is
    # recoverable from the audit trail. Cache reads bill at ~0.1x base input
    # and writes at ~1.25x (5-minute TTL), so a run whose documents are cached
    # costs a small fraction of what input_tokens alone implies. Folding them
    # together — as this client originally did — makes the JSONL unable to
    # answer "what did this run cost?" after the fact. Defaulted to 0 for
    # clients that don't report them.
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # The API's stop_reason ("end_turn", "max_tokens", ...). Carried so the
    # runner can tell a model that *answered wrongly* from one that was *cut
    # off* mid-answer — scoring those the same silently turns a budget artifact
    # into a claim about the representation. Empty when the client doesn't
    # report one. See the truncation note in the module docstring.
    stop_reason: str = ""
    # The reasoning parameters this request actually sent, echoed back so the
    # audit trail records the regime instead of assuming it. Both are "" today,
    # meaning "not sent" — which resolves to an actual regime only through
    # default_thinks(model), so a row needs BOTH these and `model` to be
    # interpretable. Carried from the client rather than filled in by the runner
    # so a future client that does send them stays honest for free.
    effort: str = ""
    thinking: str = ""


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
        # No thinking, effort, or sampling parameters — see the module docstring
        # for why each is absent and why none of them is safely tier-portable.
        content = list(self._document_blocks(document))
        content.append({"type": "text", "text": prompt})
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        # usage.input_tokens is the UNCACHED remainder only. Total input the
        # request covered adds the cached portions back; the components are also
        # carried separately so the audit trail can price them at their real
        # rates rather than all at full input price.
        usage = resp.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return AnswerResult(
            text=text,
            input_tokens=usage.input_tokens + cache_read + cache_creation,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            stop_reason=getattr(resp, "stop_reason", "") or "",
            # Neither parameter is in the create() call above, so both are ""
            # ("not sent"). Set these from the request if that ever changes —
            # they are what makes a row's regime readable after the fact.
            effort="",
            thinking="",
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
    at ~4 chars/token so the accounting paths still exercise real numbers.

    A responder may return `(text, stop_reason)` instead of a bare string to
    script a truncated response; a bare string means "end_turn". It may also
    raise instead of returning: the exception propagates out of answer() exactly
    like an SDK error from the real client — how tests script API failures such
    as a context-window overflow."""

    def __init__(self, responder):
        self._responder = responder
        self.calls: list[tuple[str, str]] = []  # (model, rendered prompt) for assertions

    def answer(self, *, model, system, prompt, max_tokens=512, document=None) -> AnswerResult:
        full = _render(document) + prompt
        self.calls.append((model, full))
        reply = self._responder(model, system, full)
        text, stop_reason = reply if isinstance(reply, tuple) else (reply, "end_turn")
        return AnswerResult(
            text=text,
            input_tokens=max(1, (len(system) + len(full)) // 4),
            output_tokens=max(1, len(text) // 4),
            stop_reason=stop_reason,
        )
