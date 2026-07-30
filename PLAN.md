# Plan to finish DecantCC

**Goal.** Find representations that cut input tokens without losing meaning for
an LLM reader — and where meaning *is* lost, price the loss, so the trade is a
decision rather than a guess.

**Done means:** a token-vs-accuracy frontier per case, with the reliability
spread attached to every point, so each lever reads as *"−X% tokens, −Y points
of weak-reader accuracy, spread moved Z"* — plus a recommended representation
defended against a held-out question set.

The binding constraint is **weak-model (Haiku) accuracy**, not strong-model
accuracy. A compression only Opus can read is the treacherous-degradation
failure this project exists to prevent.

---

## State as of 2026-07-25

**Baseline of record** (`report.md`, 450 rows, 2026-07-16, gitignored):

| arm | Haiku | Opus | strong tok | spread |
|---|---|---|---|---|
| decant-plain | 0.95 | 1.00 | 44,031 | +0.05 |
| decant | 0.95 | 1.00 | 63,665 | +0.05 |
| raw | 0.83 | 0.98 | 121,139 | +0.15 |
| markitdown | 0.83 | 0.93 | 52,209 | +0.10 |
| docling | 0.82 | 0.83 | 51,249 | +0.01 |

Contamination: zero. **But this baseline is stale** — it scored 45 questions
and the banks now hold 52, and `messy-scan`'s decant arms were regenerated
after it (`6ced6bd`). Every other conversion is byte-intact, so most of those
450 rows remain reusable via `--resume`.

**These rows are clean — the regime warning on them is a provenance gap, not a
defect.** Every trail on disk predates regime recording, so a regenerated
report now carries _"N row(s) predate regime recording and assert nothing about
how they were measured."_ That is accurate about the *file* and wrong as an
impression of the *data*: all five trails are `claude-opus-4-8` / `haiku-4-5`
only, both thinking-off by default, so every row was measured under the one
regime the harness has ever used. The warning stays as-is deliberately — it
says what the file can prove rather than what we happen to know, and softening
it would mean trusting exactly the assumption that stopped holding on Opus 5.
Rows written from here carry the fields and the warning goes away on its own.

**Three measurement problems block the reduction phase:**

1. **Saturation.** 27 of the 45 baseline questions are perfect on every arm ×
   every model. `private-novel` and `public-famous` are entirely at ceiling.
   A bank that is 60% ceiling reads "no loss" no matter what you cut.
2. **No dev/test split.** No `split` field, no flag. The whole reduction phase
   is "iterate against questions", which is exactly how you converge on keeping
   only the asked-about facts.
3. **No weak-only runs.** `cli.py` hardcodes `models = [strong, weak]`. Since
   Haiku accuracy is the binding constraint, loop iterations should skip Opus —
   the difference between a ~$2 and a ~$8 lever test, times every lever.

**Measured noise floor** (3× messy-scan run): 2 of 120 cells flip across three
draws (~2%). One draw is fine for *ranking converters*; on a 10–15 question
case a single flip is 7–10 accuracy points. **Deltas under ~5pp are
unresolvable at `--repeats 1`** — which is most of what the reduction phase is
looking for.

---

## Decisions taken 2026-07-25

**`clean-text-image-only` retired.** It existed to exercise the OCR arm, and
its `questions.json` was a verbatim copy of `clean-text`'s — duplicate question
ids, no Decant arm, no independent signal about the codec. Record preserved in
`docs/experiment-image-only-ocr.md` and `report-clean-text-image-only.md`.

**`unlimited-ocr` rescoped to `messy-scan`.** It stays as a diagnostic, not a
general arena arm. `table-heavy` has zero broken-font pages, so OCR can never
be the sole carrier of an answer there; scoring it made the arm span 2 of 6
cases and skewed the aggregate. Its output moved to
`corpus/table-heavy/experiments/`.

**Does better OCR help on `messy-scan`? The current questions cannot tell us.**
All 10 land on native-text pages or on figures no arm can read — `markitdown`,
a naive parser, scores 7/10 on Opus, which proves the answers were never on the
hard pages. The case has 32 broken-font pages (Tables 1–12) and 20 image-only
pages, and **zero questions probe them**. Measured, OCR looks worthless
(0.60/0.73 @138K vs docling 0.60/0.70 @115K — one third of one question for
+23K tokens). Grepped, it is the *only* arm holding that content:

| gold | decant | decant-plain | unlimited-ocr | docling | markitdown |
|---|---|---|---|---|---|
| 18.71 (Table 11) | 0 | 0 | **2** | 0 | 0 |
| 2,760 (Table 1) | 0 | 0 | **2** | 0 | 0 |
| 4.32 (Table 13, scan) | 0 | 0 | **3** | 1 | 0 |

**The reframe: OCR is a missing stage in Decant's pipeline, not a rival
converter.** Decant doesn't garble those pages — it declares them:
`[this page's text could not be decoded — its fonts carry no readable character
map … the page's content is in the attached figure]`. Honest, but it means the
only way Decant conveys ~52 of that document's pages is the image companion,
the most expensive object in the corpus (raw `messy-scan` is 213K tokens and
overflows Haiku's window entirely), and `decant-plain` — the 44K-token cost
champion — doesn't carry that content at all. The retired image-only
experiment points the same way: Decant 0.3.1 exits `passthrough (no-text)` on a
text-free PDF, while OCR scored 1.00/1.00 at 29K tokens against raw vision's
0.88/1.00 at 65K. Hence **lever #1 below**.

---

## Phase A — close the measurement leaks (offline, free) — **DONE 2026-07-25**

1. ~~Merge `fix/grader-repeats-cost-accounting` to `main`.~~ Done; merged with
   PR #5's failed-call `status` work, which had landed on main in parallel.
2. ~~`regrade` the 450-row trail against today's questions.~~ Done. The
   2026-07-16 baseline **disagrees with today's questions and graders**, so it
   cannot be reused wholesale as the definitive baseline.
3. ~~Fix the known grader false negatives.~~ Done — and the diagnosis was
   wrong in a way that mattered. `well-depth-difference` was already fixed by
   `2bf3373`, and the "bold-rule swing" was a symptom, not the defect. The real
   defect: **a prose answer restates context before the value, so the first
   number is a year or a section label** — "for 2025, CERN employed 808
   technicians" graded 2025; "Scope 1 GHG emissions … were 3,782,020" graded 1.
   22 of 304 numeric rows were affected and **every one was a weak-tier row**,
   so the bug was suppressing weak-model accuracy and inflating the reliability
   spread — the headline metric. Fixes, both of which *reduce* over-crediting:
   - numbers introduced by a date or label word are discounted when another
     number is available (a year-valued gold still grades, and the
     `"800.00, not 1250.00"` negation guard is intact);
   - an answer opening with the harness's own `NOT FOUND` convention is a
     decline, so no grader mines the explanation that follows for a match —
     this removed 6 rows that had been credited for anchors appearing in the
     model's account of what it *couldn't* find;
   - plus a genuine regression found by measurement, not by the plan: the `:`
     commitment-marker rule read the **ratio colon** in `"2.31, or 2.31:1"` and
     graded the answer as 1, costing five strong-tier rows.

   Net effect on the 2026-07-15 trail: 20 rows up, 6 down, all weak-tier.
4. ~~Add `split` + `--split`.~~ Done: `dev | test | retired | ""`, with
   `--split` on both `run` and `regrade`. `all` (default) = everything not
   retired, so retiring a saturated question actually stops it costing money.
5. ~~Add a weak-only run mode.~~ Done: `--weak-only`. The report ranks and
   prices on the tier that ran and reports **no** spread, rather than the
   `+0.00` that a single-tier run would otherwise produce and that would read
   as "perfectly robust".

124 tests green.

## Phase B — finalize questions

6. Verify the drafted golds in `corpus/question-candidates-2026-07-24.md`
   against `source.pdf` and merge. **The 5 messy-scan recognition questions are
   the priority** — without them the OCR question stays unanswerable at any
   price. Four are OCR-answerable; `d6-bromide-january` is the vision-tier
   probe OCR garbles, which is what tells you whether vision still beats OCR.
   (The instruction to re-copy `clean-text` questions into
   `clean-text-image-only` is void — that case is retired.)
7. Park the 27 saturated questions (`"split": "retired"`); keep them in the
   files as the record.
8. Retro-tag untagged questions with `source`. `chart-heavy` has zero tags and
   `table-heavy`'s 12 non-figure questions have none, so the per-figure slicing
   already built cannot see them.
9. Decide `border-application-areas` (figure-2) and `max-water-depth-diff`
   (figure-6): all-zero on every arm including raw vision — no signal at any
   price. Simplify to a single-read version or retire.
10. Split dev/test, then **freeze**. The test bank is not touched again until a
    candidate is being declared done.

## Phase C — the definitive baseline (one billed run, then locked)

11. Offline dry run against the fakes; gold-uniqueness grep; confirm no arm is
    missing a file.
12. One run: `--repeats 3`, all arms, control on, `--resume` off the audited
    rows where still valid. **Est. $20–35** (estimate × 0.29 for cache reads).
13. Tag the commit and **commit the report + JSONL** — they are gitignored
    today; the definitive one belongs in the repo as the anchor.
14. Report the `messy-scan` raw/Haiku context overflow (201K > 200K) as a *fit*
    failure, separately from transfer failure. It currently does a lot of the
    work in raw's +0.15 spread.

## Phase D — the reduction loop

Each lever is a new file in `conversions/`; `--resume` means only the new arm
is billed. Loop: build arm → weak-only dev run (~$1–2) → keep Pareto
improvements only → survivors get a full strong+weak dev run → held-out test
run at the end.

1. **`decant+ocr`** — decoded text where fonts decode, OCR text where they
   don't, in place of "unreadable, see figure". Tested against the new
   recognition questions. Potentially the largest single Pareto win in the
   corpus: it could let `decant-plain` carry 52 pages it currently drops, and
   let `decant` drop companion pages it currently needs.
2. **Page furniture / boilerplate removal** — cheapest, lowest risk.
3. **Table re-encoding** — markdown tables are token-expensive; CSV/TSV, drop
   empty columns, transpose wide tables.
4. **Figure-representation tiers** — companion page vs text-flattened vs
   `[omitted]`, resolved per figure using the `source` tags.
5. **Extractive prose pruning** — last, and extractive only. Abstractive
   paraphrase is dangerous; extend Decant's honesty-marker convention if used.

## Reporting gap

The report emits a ranking table. The frontier — Δtokens vs Δweak-accuracy with
the spread guard — is the deliverable and needs its own section.
