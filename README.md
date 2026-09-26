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
   `ANSWER:`), take the answer's *first* value number (not a label, year, or
   negated number), match set items on word boundaries outside negations,
   charge list answers for extra items they guess, treat `long-term` and
   `long term` alike, route anything wordier to the judge, and never
   keyword-guess an unparseable verdict. See `grading.py`.
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

`questions.json`: a list of `{id, question, gold, type, tolerance?, source?,
split?, sign_insensitive?}`, where `type` ∈ `numeric | exact | set |
ordered_list | open`. `sign_insensitive` (numeric only) grades the magnitude,
for reductions that prose states as either `21.4%` or `-21.4%`. The loader
rejects duplicate ids, unparseable numeric golds, negative tolerances and
non-list set golds.
`source` tags where the answer lives (`figure-9`, `table-3`, `text`) and drives
the report's per-source slice. `split` ∈ `dev | test | retired` selects the
question's bank — see `--split` below. Both are optional; see
`corpus/questions.schema.json`.

## Run

```bash
# Offline — the whole suite, stdlib only, no API key (from repo root):
python -m unittest discover tests

# Real run (needs Anthropic credentials):
pip install -r requirements.txt
python -m decant_eval.cli run --corpus ./corpus \
    --strong claude-opus-4-8 --weak claude-haiku-4-5 --out report.md
```

Rows stream to a JSONL sidecar (`--rows`, default `<out>.jsonl`) as they
complete, so a crash mid-run loses nothing and the graded-testing pass has a
per-answer audit trail; `--resume` continues from it without re-billing done
rows. Each row records content hashes of the document, question and gold it
was measured against, so a resumed run re-asks cells whose conversion or
question changed, and the report counts only rows for the current corpus,
split, arms, models and repeats (the rest stay in the file and are counted in
a "Rows not scored" section). A fresh run refuses a non-empty rows file —
resume it or pick a new `--rows`. The control arm is written to the same
trail and resumes too; judge calls are costed on the row. The document sits in its own `cache_control` block, so the loop re-uses it
across every question about it (~90% cheaper on large documents). Flags:
`--no-raw` skips the source-PDF baseline, `--no-control` skips the control arm,
`--repeats N` samples each question N times (sampling can't be pinned on the
strong tier, so one draw per cell is a draw, not a measurement).

The scoreboard compares arms only on cases where every arm appears, so a
diagnostic arm present in one case would shrink it to that case. A run whose
common cases would be under half the corpus is refused before billing: pass
`--exclude-arm NAME` (repeatable; also on `regrade`) to leave that arm out, or
`--allow-partial-arms` to run anyway. Dated snapshot IDs
(`claude-haiku-4-5-20251001`) are accepted and classify as their alias; pin
them for a locked baseline.

Two flags exist for the compression phase, where the job is to cut tokens
without losing meaning:

- `--split dev|test|all|retired` picks the question bank. Tune a compression
  candidate against `dev` and keep `test` held out until declaring it done —
  iterating against the same questions you report on converges on keeping only
  the asked-about facts. `all` (the default) is everything not `retired`;
  retiring a question that every arm answers perfectly stops it costing money
  for no discrimination.
- `--weak-only` runs the weak model alone. Weak-reader accuracy is the binding
  constraint — a compression only the strong model can read is the treacherous
  degradation this project exists to prevent — so loop iterations don't need
  the strong tier. The report then ranks and prices on the tier that ran and
  reports no spread, rather than implying one it never measured.

`python -m decant_eval.cli regrade --corpus DIR --rows FILE.jsonl --out OUT.md`
re-grades an existing audit trail against the current graders and questions.
Free by default (`numeric`/`set`/`ordered_list` never consult the judge;
`exact`/`open` keep their stored verdict unless `--judge MODEL` is passed, which
is billed). It never modifies the input file, and refuses an `--out-rows` that
would. This is what makes a grader fix
applicable to answers already paid for, and it repairs the resumed-run trap
where `--resume` carries rows graded under an older grader.

`tests/fixtures/sample-invoice/` is a worked example case (clean vs.
deliberately garbled conversion) and the harness's end-to-end fixture. It
lives with the tests rather than in `corpus/` so a real run never scores it
or lets its test-only arms break the report's common-case comparison.
