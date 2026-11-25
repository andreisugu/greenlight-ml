# Methodology and Current Evidence Status

This document records the data population and methodological gates for GreenlightML. It separates what the CSV files show from assumptions about commercial outcomes.

## 1. Prediction setup (V1)

The decision is pre-release film assessment. The dataset contains a later TMDB snapshot, so each candidate predictor needs evidence that it existed at the decision date. A CSV column being present does not establish historical availability.

The V1 settings are:

- **Primary target:** `revenue > budget` (`target_nominal`, 75.50% prevalence).
- **Sensitivity target:** `revenue >= 2 * budget` (`target_hollywood`, 56.02% prevalence).
- **Population:** 3,229 titles with positive finite reported budget and revenue.
- **Split:** random 80/20 stratified, `random_state=42`; 2,583 train / 646 test.
- **Features:** 29 basic pre-release features (log budget, runtime, release year/month, 19 genres, top-5 languages + Other).
- **Excluded:** `popularity`, `vote_average`, `vote_count`; synopsis TF-IDF and director/cast encoding are V2 scope.

The feature list is pre-release by *intent*, not by audited provenance: the per-column availability gate in the EDA report (limitation 5) is still open, so "pre-release" here does not yet mean "proven to exist at a fixed decision date".

Neither label is verified studio net profitability: the data lacks exhibitor settlements, marketing/distribution, financing, and other recoupable costs. V1 reports the nominal target as primary and the 2× target only as sensitivity.

## 2. Primary EDA population and selection limits

TMDB 5000 has 4,803 merged rows. The current dump contains 1,037 zero budgets and 1,427 zero revenues; 1,574 rows have at least one zero. We treat these zeros as *unreported financials* for this analysis, not as proof of failure or lack of theatrical release.

The primary descriptive/modeling candidate population is the **3,229 rows with positive finite budget and revenue**. On this selected population:

- `revenue > budget`: 2,438 rows (75.50%).
- `revenue >= 2 * budget`: 1,809 rows (56.02%).

These are conditional prevalences among records with reported financials. They should not be generalized to all films without addressing selection/missingness.

### $10K threshold sensitivity

A strict `budget > 10,000 AND revenue > 10,000` rule leaves 3,203 rows, but it is **not an anomaly detector**. It excludes 26 records: 16 with `revenue <= budget` and 10 with `revenue > budget`. The CSV values alone do not establish that every excluded row is corrupt; known microbudget examples appear in this set. The threshold may be reported as a sensitivity scenario, never silently substituted for the primary population.

### Temporal regime and entity structure

The population shows the label is **not stationary**: the `revenue > budget` rate runs from **91.6% before 1980** to **74.1%
from 2000 onward**, while the median reported budget grows from **$0.4M to $30M**. It also shows the rows are **not
independent**: 628 directors and 1,425 lead actors recur across retained titles. V1 uses the random stratified split
and therefore claims only this holdout, not future-film generalization. Chronological
and grouped validation are explicit later-stage audits, not optional silent alternatives. All figures:
[`docs/DATA_DICTIONARY.md`](./DATA_DICTIONARY.md).

## 3. Evaluation protocol and V1 status

V1 follows the protocol below and publishes the first accepted model receipt: [`notebooks/02_baseline_linear_model.ipynb`](../notebooks/02_baseline_linear_model.ipynb) and [`docs/V1_REPORT.md`](./V1_REPORT.md). The earlier benchmark remains withdrawn: it fit TF-IDF vocabulary and category-frequency decisions before the train/test split and used the disputed >$10K population restriction. The leakage bug is fixed (`src/features.py` exposes a train-fit `FeatureEncoder`; `scripts/benchmark.py` splits raw rows before fitting), and V1 puts the encoder and scaler inside a `Pipeline` so they are refit inside every CV fold.

Before reporting model performance:

1. Define one target proxy, population, prediction date, and unit of observation.
2. Choose a split that matches the claim (chronological for future-film generalization; grouped if related observations can cross splits).
3. Fit imputers, scalers, vocabularies, category selection, feature selection, and tuning **inside training folds only**.
4. Keep a final holdout untouched until model and threshold choices are frozen.
5. Compare against a transparent baseline on the same split.
6. Report prevalence, balanced metrics, uncertainty/variation, and error analysis—not raw accuracy alone.

V1 fixes the target proxy, population, unit of observation, split, and train-only preprocessing, so steps 2–5 hold for this holdout. Step 1 is only partially closed: the prediction date and the per-column availability of the pre-release features are assumed, not audited. Step 6 is partially met — prevalence, balanced metrics, and a confusion matrix are reported, but variation is the 5-fold sweep rather than a confidence interval. V2 repeats the protocol with text and people features and reaches ROC-AUC 0.709 / PR-AUC 0.874 on the same holdout ([`V2_REPORT.md`](./V2_REPORT.md)); notebook 04 remains the unexecuted scaffold for the final audit.

## 4. Leakage and temporal provenance

`popularity`, `vote_average`, and `vote_count` are leakage candidates for a pre-release decision because they reflect audience activity/ratings. The empirical correlations in the EDA are descriptive and are **not** the definition of leakage. Establish the timestamp/availability of each predictor relative to the forecast decision, then compare identical train/test protocols with and without post-release fields.

## 5. Financial metric terminology

The repository's gross-minus-budget calculation is a **gross-to-budget surplus proxy**, not net portfolio ROI. It assumes all reported box-office gross is available to repay production budget and omits exhibitor shares and other costs. Any financial interpretation must state this limitation; do not claim realized studio returns.
