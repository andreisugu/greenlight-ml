# Metrics and Financial Proxies

## Classification metrics

The primary population contains 3,229 rows with positive finite reported budget and revenue. On it, the nominal gross-to-budget label `revenue > budget` has prevalence 75.50%; the alternative `revenue >= 2 * budget` label has prevalence 56.02%.

These are selected-sample label prevalences, not proof of studio profitability.

### What the names mean

| Term | Plain meaning | "No skill" value |
|---|---|---|
| Accuracy | Share of titles labelled correctly | 0.755 (always "hit") |
| Balanced accuracy | Average of hit-recall and miss-recall | 0.500 |
| Precision | Of the titles called hits, the share that are | 0.755 |
| Recall | Of the real hits, the share found | 1.000 (always "hit") |
| Macro F1 | Precision and recall combined, both classes weighted equally | 0.430 |
| ROC-AUC | Chance a random hit outranks a random miss | 0.500 |
| PR-AUC | Same idea, focused on the titles selected | 0.755 |
| Threshold | Probability above which a title is called a hit | — |
| Cross-validation | Score each of 5 parts with a model trained on the other 4 | — |
| OOF | "Out-of-fold": a training row scored by a model that did not see it | — |

"No skill" is what a predictor that always answers "hit" scores on the same 646-title holdout, so it is the number a
model has to beat. V1 reports all of these on that holdout; the full table is in [`V1_REPORT.md`](./V1_REPORT.md).

## Financial proxy (not ROI)

The current arithmetic

```text
(sum(revenue) - sum(budget)) / sum(budget)
```

is named **Gross-to-Budget Surplus Proxy**. It is a gross box-office multiple less one, weighted by production budget. It is not net ROI or studio cash return: it does not account for exhibitor shares, Print & Advertising, distribution, financing, or other costs. The 2× label is likewise a rough target proxy, not an accounting result.

V1 does not publish a portfolio or ROI number. The gross-to-budget proxy is computed by `src/metrics.py` but is not used as V1 evidence.

## Status of historical benchmark values

Previously published benchmark receipts (including the 641-title holdout and ROI uplift claims) are withdrawn as validated evidence. They came from a pipeline that fit feature vocabularies/category choices before the train/test split, and they used a disputed >$10K population restriction. The leakage bug is fixed (`FeatureEncoder` + split-before-fit), and V1 now fixes its settings and publishes a receipt. The remaining 03–04 notebooks are still drafts.
