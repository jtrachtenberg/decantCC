# DecantCC — LLM-digestibility eval harness

Measures how well a document **conversion** transfers meaning to an LLM reader,
for the fewest tokens. It is the "psychovisual model" for the decantCC idea
(JPEG-for-LLMs: a reliable, not pixel-accurate, document representation for LLM
endpoints) — the measuring instrument you build *before* the format. The same
corpus and questions serve both the "grade the conversion + my confidence"
testing pass and the companion-successor exploration.

## What it does

For every **case** (a source document), it feeds each candidate **conversion**
to a strong and a weak target model, asks the case's questions, grades the
answers, and reports three things per conversion:

- **accuracy** per model tier — did the reader get the answer right?
- **cost** — mean input tokens (the conversion's token weight)
- **reliability spread** = strong accuracy − weak accuracy. The novel metric: a
  conversion that lets the *weak* model answer as well as the strong one
  transfers meaning robustly rather than relying on a smart reader to recover
  from a bad representation. **Lower is better.**

It scores **cached conversion outputs** — plain `.md`/`.txt` files — and never
runs the converters. So it's engine-agnostic and GPU-free: Decant, MarkItDown,
Docling, olmOCR, or a future decantCC representation is just another file in a
case's `conversions/` folder.

## The four methodology decisions it rests on

1. **Grading — hybrid.** Discrete answers grade programmatically (numeric /
   exact / set); genuinely open answers fall back to an LLM judge. A grader
   that can itself be wrong would reintroduce the treacherous-degradation
   problem the eval exists to measure, so programmatic is preferred wherever
   the question allows. The graders **must not over-credit hedged, negated, or
   verbose answers** — the exact answers a model produces when reading a
   *corrupted* conversion — so they grade a short answer (final line / explicit
   `ANSWER:`), take the answer's *first* number, match set items on word
   boundaries, route anything wordier to the judge, and never keyword-guess an
   unparseable verdict. See `grading.py`.
2. **Endpoints — strong + weak** (`claude-opus-4-8` + `claude-haiku-4-5`), for
   the reliability spread. *Caveat:* this measures answer quality via the API;
   platform image-token billing is a separate accounting model.
3. **Arena — raw upload + Decant + MarkItDown + Docling** (extensible), so the
   ranking answers "is this better than what exists?", not just A/B tuning. The
   **raw upload** anchor is the source PDF itself, fed as a document block (not
   extracted text) — the baseline every conversion is measured against.
4. **Authoring — drafted from the source, verified by a human.** Gold answers
   come from the original document, **never** from a conversion — otherwise the
   eval tests conformance to a converter, not correctness. A **no-document
   control arm** runs each question with no document at all; any question the
   model answers correctly from memory is flagged, so a famous document can't
   flatter every conversion (and it doubles as the contamination audit).

*Thinking is off* for the target models: the harness measures whether the
representation carries the meaning, not whether a model can reason around a
corrupted one (and it sharpens the spread). A knob to revisit.

## Layout

```
corpus/<case>/
  source.pdf              # reference only — never fed to a model
  questions.json          # or questions.yaml (needs pyyaml)
  conversions/
    raw.md  decant.md  markitdown.md  docling.md
```

`questions.json`: a list of `{id, question, gold, type, tolerance?}`, where
`type` ∈ `numeric | exact | set | open`.

## Run

```bash
# Offline — the whole suite, stdlib only, no API key:
cd eval && python -m unittest discover tests

# Real run (needs Anthropic credentials):
pip install -r requirements.txt
python -m decant_eval.cli run --corpus ./corpus \
    --strong claude-opus-4-8 --weak claude-haiku-4-5 --out report.md
```

Rows stream to a JSONL sidecar (`--rows`, default `<out>.jsonl`) as they
complete, so a crash mid-run loses nothing and the graded-testing pass has a
per-answer audit trail; `--resume` continues from it without re-billing done
rows. The document sits in its own `cache_control` block, so the loop re-uses it
across every question about it (~90% cheaper on large documents). Flags:
`--no-raw` skips the source-PDF baseline, `--no-control` skips the control arm.

`sample-invoice/` is a worked case (clean vs. deliberately garbled conversion)
and doubles as the harness's end-to-end fixture.
