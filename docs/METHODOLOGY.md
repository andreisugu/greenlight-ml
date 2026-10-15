# Methodology and Current Evidence Status

This document records the data population and methodological gates for GreenlightML. It separates what the CSV files show from assumptions about commercial outcomes.

## 1. Prediction contract (still to be finalized)

The intended decision is pre-release film assessment. The dataset contains a later TMDB snapshot, so each candidate predictor needs evidence that it existed at the decision date. A CSV column being present does not establish historical availability.

Two labels are currently defined as **gross-to-budget proxies**:

- Nominal proxy: `revenue > budget`.
- Alternative 2× proxy: `revenue >= 2 * budget`.

Neither is verified studio net profitability. The data does not include exhibitor settlements, marketing/distribution costs, financing, or all recoupable costs. The project must choose the primary proxy and report the other as sensitivity analysis.

## 2. Primary EDA population and selection limits

TMDB 5000 has 4,803 merged rows. The current dump contains 1,037 zero budgets and 1,427 zero revenues; 1,574 rows have at least one zero. We treat these zeros as *unreported financials* for this analysis, not as proof of failure or lack of theatrical release.

The primary descriptive/modeling candidate population is the **3,229 rows with positive finite budget and revenue**. On this selected population:

- `revenue > budget`: 2,438 rows (75.50%).
- `revenue >= 2 * budget`: 1,809 rows (56.02%).

These are conditional prevalences among records with reported financials. They should not be generalized to all films without addressing selection/missingness.

### $10K threshold sensitivity

A strict `budget > 10,000 AND revenue > 10,000` rule leaves 3,203 rows, but it is **not an anomaly detector**. It excludes 26 records: 16 with `revenue <= budget` and 10 with `revenue > budget`. The CSV values alone do not establish that every excluded row is corrupt; known microbudget examples appear in this set. The threshold may be reported as a sensitivity scenario, never silently substituted for the primary population.

### Temporal regime and entity structure

The EDA shows the label is **not stationary**: the `revenue > budget` rate runs from **91.6% before 1980** to **74.1%
from 2000 onward**, while the median reported budget grows from **$0.4M to $30M**. It also shows the rows are **not
independent**: 628 directors and 1,425 lead actors recur across retained titles. Consequently, step 2 of the protocol
below is not optional — prefer chronological validation for a future-film claim and grouped validation when related
observations could cross the split. A plain random split would mix eras and could leak the same entity into train and
test. All figures: [`docs/EDA_REPORT.md`](./EDA_REPORT.md).

## 3. Evaluation protocol before performance claims

No model-performance receipt in the repository is currently accepted as validated evidence. The earlier benchmark is withdrawn: it fit TF-IDF vocabulary and category-frequency decisions before the train/test split, and it used the disputed >$10K population restriction. That preprocessing bug is now fixed—`src/features.py` exposes a train-fit `FeatureEncoder` and `scripts/benchmark.py` splits the raw rows before fitting—but the target proxy, population, and decision contract are not frozen, so no scores are published yet.

Before reporting model performance:

1. Define one target proxy, population, prediction date, and unit of observation.
2. Choose a split that matches the claim (chronological for future-film generalization; grouped if related observations can cross splits).
3. Fit imputers, scalers, vocabularies, category selection, feature selection, and tuning **inside training folds only**.
4. Keep a final holdout untouched until model and threshold choices are frozen.
5. Compare against a transparent baseline on the same split.
6. Report prevalence, balanced metrics, uncertainty/variation, and error analysis—not raw accuracy alone.

The current 02–04 notebooks are unexecuted scaffolds that already use the train-fit encoder, but they are still drafts. The historical README benchmark table is not model evidence.

## 4. Leakage and temporal provenance

`popularity`, `vote_average`, and `vote_count` are leakage candidates for a pre-release decision because they reflect audience activity/ratings. The empirical correlations in the EDA are descriptive and are **not** the definition of leakage. Establish the timestamp/availability of each predictor relative to the forecast decision, then compare identical train/test protocols with and without post-release fields.

## 5. Financial metric terminology

The repository's gross-minus-budget calculation is a **gross-to-budget surplus proxy**, not net portfolio ROI. It assumes all reported box-office gross is available to repay production budget and omits exhibitor shares and other costs. Any financial interpretation must state this limitation; do not claim realized studio returns.
