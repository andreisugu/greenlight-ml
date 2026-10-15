# EDA Report — Data-Quality Audit and Analysis Population

> Written companion to [`notebooks/01_exploratory_data_analysis.ipynb`](../notebooks/01_exploratory_data_analysis.ipynb).
> Every number below is reproduced by an executed cell in that notebook; nothing here is asserted without a receipt.

This report records the Stage-1 deliverable (`01`): the schema and data-quality audit, the definition of the analysis
population, the target proxies, and the exploratory findings that constrain the later modeling stages. It is organized as
*question → method → finding → decision* so each claim carries both a method and a consequence.

---

## 1. Reproduce it

```bash
python scripts/download_data.py          # fetch the two CSVs into data/
jupyter lab notebooks/01_exploratory_data_analysis.ipynb   # Restart & Run All
python -m unittest discover -s tests -v  # data-contract, feature, and metric tests
```

The notebook is executed with a clean kernel and contains no pre-computed outputs that the code cannot regenerate.

---

## 2. Relational and schema contract

**Question.** Do the two source files join one-to-one into a unique, predictable schema?

**Method.** Assert the key cardinalities, the cross-file title agreement, the merged shape, and the positional index; the
same rules are exercised as unit tests in `tests/test_data.py`.

**Finding.** `movies` is 4,803 × 20 and `credits` is 4,803 × 4. `movie_id` duplicates `id` and `title` is identical
across the two files, so the merged schema is **4,803 rows × 22 unique columns** (not 24). The merge is validated
`one_to_one`; any unmatched key or title mismatch raises instead of silently dropping rows.

**Decision.** Treat the merge as a hard contract. A regression that breaks it fails loudly rather than quietly
shrinking the dataset.

---

## 3. Missingness, zeros, and the analysis population

**Question.** How does the dump encode unavailable financial values, and what population can we analyze honestly?

**Method.** Separate zeros, blanks, non-finite, and negative values per column; then reconcile the exclusions by channel.

**Finding.** Budget and revenue contain **no** NaN, non-finite, or negative values — only zeros. Combined with the earlier
columns, the exclusions reconcile exactly:

| Reason | Rows |
|---|---:|
| `budget = 0`, revenue reported | 147 |
| `revenue = 0`, budget reported | 537 |
| `budget = 0` **and** `revenue = 0` | 890 |
| missing / non-finite financial value | 0 |
| negative financial value | 0 |
| **Excluded (total)** | **1,574** |

**4,803 raw − 1,574 excluded = 3,229 rows** with positive, finite reported budget and revenue. Among retained rows,
3,228 are `Released` and 1 is `Post Production`.

**Decision.** The *reported-financials population* is **3,229 rows**. A zero is treated as an unreported financial value,
never as commercial failure and never imputed as profit. Selection-by-reporting is carried forward as an explicit
limitation: excluded films may differ systematically in scale or outcome, and this file alone cannot say how.

---

## 4. The >$10K threshold is a sensitivity scenario, not an anomaly detector

**Question.** Does a strict `budget > $10K AND revenue > $10K` rule isolate corrupt data, or just redefine the sample?

**Method.** Apply the threshold as a separate filter, list the removed rows, and report the label rate both ways.

**Finding.** The rule keeps **3,203 rows** and removes **26** (10 nominal gross-above-budget, 16 gross-at/below-budget).
The removed set mixes hand-verified microbudgets (*Primer*, $7,000; *Tarnation*, $218; *Eraserhead*, exactly $10,000)
with genuinely suspicious values, so it cannot be labelled uniformly erroneous.

**Decision.** The 3,229-row reported-financials population stays primary; the 3,203-row subset is reported **only** as a
sensitivity scenario. It is never called a "clean" or "canonical" dataset.

---

## 5. Target proxies

**Question.** What label can we derive from the reported money, and how far is it from real profit?

**Method.** Derive two gross-to-budget proxies and report the strict/`>=` boundary for each.

| Receipt | Primary (3,229) | Sensitivity (3,203) |
|---|---:|---:|
| `revenue > budget` | 2,438 (75.50%) | 2,428 (75.80%) |
| `revenue >= 2 × budget` | 1,809 (56.02%) | 1,800 (56.20%) |
| `revenue > 2 × budget` (strict) | 1,804 (55.87%) | 1,796 (56.07%) |

**Finding.** The majority-label baseline on the primary population is **75.50%**, so the nominal proxy is strongly
imbalanced.

**Decision.** Both proxies are gross-to-budget labels, **not** studio net profit — the files omit exhibitor settlements,
marketing/distribution, financing, and other recoupable costs. The boundary convention is reported explicitly (55.87%
vs 56.02%) instead of being hidden behind a single figure. The final frost of one proxy as *the* target is a Stage-2
(modeling) decision.

---

## 6. Distributions, concentration, and transforms

**Question.** How skewed are the money columns, and does a log transform actually help?

**Method.** Compare raw and `log1p` skewness in both populations and measure revenue concentration.

**Finding.**

| Population | Budget skew (raw → log1p) | Revenue skew (raw → log1p) |
|---|---:|---:|
| Primary | 2.09 → **−3.11** | 3.82 → **−2.35** |
| >$10K sensitivity | 2.08 → −1.10 | 3.81 → −1.13 |

`log1p` removes the right skew but **overshoots into a heavy left tail** (small budgets near $1 become extreme negative
outliers in log space). Revenue is highly concentrated: the top 5 titles are 2.34% of gross and the top 20% of titles are
65.8% of gross. Runtime has a median of **104 min** (>0, raw) and **107 min** in the primary population, where its zeros
and NaNs are already gone.

**Decision.** `log1p(budget)` is a *candidate* transform to evaluate inside the train-fit encoder, not an assumed fix.
Concentration means mean-based summaries are fragile, so medians are reported alongside means throughout.

---

## 7. Category cardinality, repetition, and structure

**Question.** Are categories balanced, and are the rows independent observations?

**Finding — cardinality.** There are **19 genre labels** (membership 5–1,441; `Drama` 1,441, `Comedy` 1,110) with two
rare genres (`Documentary` 38, `Foreign` 5) and 1 empty genre list among retained rows. There are **27 language codes**,
with English covering **96.1%** of titles.

**Finding — repetition.** The retained rows contain **1,445 distinct directors**; **628** direct more than one title
(2,412 titles total), and **1,425 actors** appear in the top-3 billing of more than one title (7,445 title slots). The
most prolific director is Steven Spielberg (27 titles).

**Finding — structure (PCA).** On the standardised block `[log_budget, runtime, vote_average, log_vote_count]`, the first
two components retain **72.7%** of variance (PC1 0.442, PC2 0.285), and the proxy classes overlap substantially — the
task looks hard and noisy rather than cleanly separable. The block intentionally includes post-release fields to show
*why* reception data separates the classes.

**Decision.** Rare genres/languages must be handled by a **train-fitted vocabulary** (collapse to `Other`). Because
directors and stars recur, a plain random split can put the same entity on both sides, so a **group-aware split** is
required when the claim depends on it.

---

## 8. Temporal coverage and regime stability

**Question.** Is the target rate stable over time, or does it drift?

**Finding.** Release years span **1916–2016**. The nominal proxy rate is **91.6% before 1980** versus **74.1% from
2000 onward**, and the median reported budget rises from **$0.4M (1910s) to $30M (2010s)**. Monthly release counts peak
in September (386 titles) while mean/median revenue peaks in June (202.71M / 112.27M).

**Decision.** The label is **non-stationary**: it is both a different prevalence and a different dollar scale across
eras. Random splits mix incompatible regimes, so time-aware validation is preferred and the model must never train on
future years. Monthly seasonality is descriptive only — it is confounded by era and blockbuster composition.

---

## 9. Associations and feature-time leakage

**Question.** Which numeric fields track revenue — and which exist only after release?

**Finding.** Pearson/Spearman correlation with revenue: `vote_count` 0.76/0.75, `budget` 0.71/0.68, `popularity`
0.60/0.70, `runtime` 0.23/0.23, `vote_average` 0.19/0.13.

**Decision.** Correlation is descriptive and is **not** the definition of leakage. `popularity`, `vote_average`, and
`vote_count` are post-release reception signals and are treated as leakage **candidates** for a pre-release task; they
stay out of the pre-release feature set and are revisited only as a controlled leakage experiment in Stage 5.

---

## 10. Acceptance checklist

| Criterion | Evidence | Verdict |
|---|---|---|
| Relational/schema contract, tested | Contract cell (7 PASS) + `tests/test_data.py` | **PASS** |
| Missing vs zero vs impossible separated; counts reconciled | §3 removal log (147/537/890) | **PASS** |
| Threshold presented as sensitivity, not anomaly label | §4 under-floor listing | **PASS** |
| Targets named and bounded against profit | §5 receipts + `docs/METRICS_AND_ROI.md` | **PASS** |
| EDA breadth: distributions, categories, relations, groups, time | §6–§9, 12 figures | **PASS** |
| Every figure answers a written question | 13 `Question` prompts | **PASS** |
| Decisions logged (observation → hypothesis → test → action) | Notebook §14 decision log | **PASS** |
| Reproducibility | Clean kernel, executed notebook, 18 passing tests | **PASS** |
| Handoff: single explicit population; no premature model claims | §5/§8; no scores published | **PASS** |

---

## 11. Limits carried forward

1. **Selection by reporting.** The 3,229-row population is conditional on reported finances; excluded films may differ
   systematically.
2. **Proxies, not profit.** Both targets are gross-to-budget labels that ignore marketing, distribution, and exhibitor
   splits.
3. **Non-stationary label.** The proxy rate and dollar scale drift across eras; validation must respect time.
4. **Non-independent rows.** Directors and lead actors recur, so entity leakage across folds is possible.
5. **Unresolved prediction date.** The intended pre-release decision date is not yet frozen; feature availability must be
   proven per column before any model receipt is published.

Downstream plan: freeze a single target, population, and prediction date; then run the modeling notebooks on an identical
locked split with train-only preprocessing (see [`docs/METHODOLOGY.md`](./METHODOLOGY.md)).
