# Question candidates — 2026-07-24 (drafted, awaiting human gold verification)

Purpose: strengthen the corpus for the next phase — ranking token-reduction
paths without losing meaning. Every gold below was read from the **source
PDF page cited** (rendered visually; never from a conversion), but per the
hard rule each needs your human verification against `source.pdf` before it
enters `questions.json`. "OCR-answerable" notes whether the value survived
into the existing `unlimited-ocr.md` conversion (checked by grep), i.e.
whether the current OCR arm can in principle score it.

---

## messy-scan — recognition-tier questions (all in BROKEN or IMG-ONLY pages)

Page classes measured 2026-07-23: 32 BROKEN-font pages (incl. p46–59 =
Tables 1–12), 20 IMG-ONLY (incl. alternating scans p67–97), 46 native.
Parsers extract cipher garbage or nothing from all pages cited below.

```json
{
  "id": "wheat-selenium-elbert-north",
  "question": "In the wheat-grain sample data (Table 6), what was the selenium concentration, in mg/kg, for the North (control) field at the Elbert County site? Answer with the number only.",
  "gold": "2.1", "type": "numeric", "tolerance": 0, "source": "table-6"
}
```
- Source: PDF p52 (printed 44). OCR-answerable: **yes** (value + labels present).
- Expected: unlimited-ocr answers; parsers fail; raw-Opus likely answers.

```json
{
  "id": "wheat-nickel-arapahoe-middle",
  "question": "In the wheat-grain sample data (Table 6), what was the nickel concentration, in mg/kg, for the Middle (biosolids application) field at the Arapahoe County site? Answer with the number only.",
  "gold": "1.35", "type": "numeric", "tolerance": 0, "source": "table-6"
}
```
- Source: PDF p52. OCR-answerable: **yes**.

```json
{
  "id": "dtx10b-february-level",
  "question": "Per the monthly water-level data (Table 11), what was the water level (W.L. bmp, in feet) for well DTX10B in February 2000 (02/04/00)? Answer with the number only.",
  "gold": "18.71", "type": "numeric", "tolerance": 0, "source": "table-11"
}
```
- Source: PDF p57 (printed 49). OCR-answerable: **yes** (DTX10B and 18.71 both present).

```json
{
  "id": "sec3-wet-tons",
  "question": "Per Table 1, what were the total wet tons of biosolids applied to the area with legal description N 1/2 SEC 3 T6S R58W? Answer with the number only.",
  "gold": "2760", "type": "numeric", "tolerance": 0, "source": "table-1"
}
```
- Source: PDF p47 (printed 39), DC 322 row. OCR-answerable: **yes** ("2,760" and
  the legal description present; note some other columns of this row were
  dropped by OCR — verify the row binding when you check gold).

```json
{
  "id": "d6-bromide-january",
  "question": "Per the water-quality data (Table 13), what was the dissolved bromide concentration, in mg/L, for well D6 sampled on 01/11/00? Answer with the number only.",
  "gold": "4.32", "type": "numeric", "tolerance": 0, "source": "table-13-scan"
}
```
- Source: PDF p69 (printed 61) — a **scanned image-only page**. OCR-answerable:
  **no** (current OCR garbled this value) — this is the designed vision-tier
  probe: only raw vision (and future better OCR) should score it. Keep exactly
  one of these; it replaces the role figure-2/figure-6 were failing to fill.

### Recommended edits to existing messy-scan questions
- `border-application-areas` (figure-2): all 12 arm×model runs failed incl.
  raw vision → no ranking signal. Also has typos ("between between", "Give
  you answer"). Either fix typos and accept it as the case's single
  aspirational ceiling question, or simplify to a single-read version.
- `max-water-depth-diff` (figure-6): same all-zero situation (chart read +
  week-over-week arithmetic). Suggest simplifying to a one-step read, e.g.
  peak month, if you want signal from it.
- `well-depth-difference` (table-10): **keep unchanged.** It is answerable
  (raw-Opus produced 20.59/22.19 → 1.60 but flunked the first-number grader
  by showing work against the "single number only" instruction). Known
  false-negative to remember when reading reports.

---

## chart-heavy — tagged chart-only questions (currently 0 source tags in this case)

```json
{
  "id": "staff-engineers-pct",
  "question": "According to the CERN staff breakdown for 2025, what percentage of staff were engineers and applied scientists? Answer with the number only.",
  "gold": "48.55", "type": "numeric", "tolerance": 0, "source": "figure-staff-donut"
}
```
- Source: PDF p53 staff donut (1341 = 48.55%).

```json
{
  "id": "admin-office-staff",
  "question": "According to the CERN staff breakdown for 2025, how many administrators and office staff did CERN employ? Answer with the number only.",
  "gold": "472", "type": "numeric", "tolerance": 0, "source": "figure-staff-donut"
}
```
- Source: PDF p53 staff donut.

```json
{
  "id": "energy-water-expense",
  "question": "What were CERN's energy and water expenses for 2025, in MCHF? Answer with the number only.",
  "gold": "91.90", "type": "numeric", "tolerance": 0, "source": "figure-expenses-donut"
}
```
- Source: PDF p53 expenses donut (6.58% slice).

```json
{
  "id": "materials-goods-component",
  "question": "Within CERN's 2025 Materials expenses, how many MCHF were goods, consumables and supplies (as opposed to other materials expenses)? Answer with the number only.",
  "gold": "292.28", "type": "numeric", "tolerance": 0, "source": "figure-expenses-donut"
}
```
- Source: PDF p53, sub-label inside the Materials callout (530.14 = 292.28 +
  237.86). Fine-grained — the weak-reader discriminator of this batch.

```json
{
  "id": "safety-elearning-courses",
  "question": "According to the Safety Training panel, how many e-learning safety courses were followed? Answer with the number only.",
  "gold": "114000", "type": "numeric", "tolerance": 0, "source": "figure-safety-training"
}
```
- Source: PDF p41 stat panel ("114 000+"). Spreads tags beyond p53.

### Recommended retro-tags for existing chart-heavy questions (metadata only)
- `staff-2023` → `"source": "figure-staff-evolution"` (p53 table)
- `technicians` → `"source": "figure-staff-donut"`
- `personnel-expense` → `"source": "figure-expenses-donut"`

---

## clean-text — deep-prose + fine-conditional questions

These target content a prose compressor drops first (conditional deadlines,
interim-period clauses, footnotes). **After merging, re-copy questions.json
verbatim into `clean-text-image-only/`** so the A/B case stays aligned. All
four golds verified present in the image-only `unlimited-ocr.md` too, so the
A/B remains fully loaded.

```json
{
  "id": "draft-deadline",
  "question": "Where a draft is required, by what date are the DRAFT audited financial statements due?",
  "gold": "February 15, 2022", "type": "exact", "source": "text-p3"
}
```
```json
{
  "id": "final-deadline-no-draft",
  "question": "Where a draft is not required, by what date are the FINAL audited financial statements with signed opinion due?",
  "gold": "March 1, 2022", "type": "exact", "source": "text-p3"
}
```
- The pair probes the draft-required vs not-required conditional; a summary
  that collapses the two deadlines fails one of them.

```json
{
  "id": "asu-interim-periods-date",
  "question": "Under ASU 2015-03, for entities other than Public Business Entities, the amendments cover interim periods within fiscal years beginning after what date?",
  "gold": "December 15, 2016", "type": "exact", "source": "text-p4"
}
```
- The discriminator is the 2016 interim clause vs the 2015 headline date —
  exactly the micro-detail compression loses.

```json
{
  "id": "sample-partnership-name",
  "question": "What is the name of the imaginary limited partnership used for the sample financial statements?",
  "gold": "XYZ Limited Partnership", "type": "exact", "source": "text-p5"
}
```

---

## Cases deliberately left unchanged
- **table-heavy**: strongest existing set (15 questions, tags, ordered_list).
- **public-famous**: adequate for its contamination-control role.
- **private-novel**: adequate; lowest marginal value.

## Verification checklist (per the hard rule)
1. Open the cited `source.pdf` page and confirm each gold value/wording.
2. Merge approved entries into the case `questions.json` (and re-copy
   clean-text → clean-text-image-only).
3. Decide the figure-2 / figure-6 edits.
4. Next eval run then measures the recognition tier and compression
   sensitivity these questions add — no conversion regeneration needed.
