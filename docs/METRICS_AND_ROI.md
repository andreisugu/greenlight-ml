# Metrics and Financial Proxies — No Validated Model Receipts Yet

## Classification metrics

The primary EDA population contains 3,229 rows with positive finite reported budget and revenue. On it, the nominal gross-to-budget label `revenue > budget` has prevalence 75.50%; the alternative `revenue >= 2 * budget` label has prevalence 56.02%.

These are selected-sample label prevalences, not proof of studio profitability. Future model reports should include a majority-class baseline, balanced accuracy, per-class precision/recall, macro F1, ROC-AUC, confusion matrix, and the exact split/population.

## Financial proxy (not ROI)

The current arithmetic

```text
(sum(revenue) - sum(budget)) / sum(budget)
```

is named **Gross-to-Budget Surplus Proxy**. It is a gross box-office multiple less one, weighted by production budget. It is not net ROI or studio cash return: it does not account for exhibitor shares, Print & Advertising, distribution, financing, or other costs. The 2× label is likewise a rough target proxy, not an accounting result.

## Status of historical benchmark values

Previously published benchmark receipts (including the 641-title holdout and ROI uplift claims) are withdrawn as validated evidence. They came from a pipeline that fit feature vocabularies/category choices before the train/test split, and they used a disputed >$10K population restriction. The leakage bug is now fixed (`FeatureEncoder` + split-before-fit), but the target/decision contract is not frozen. Recompute and publish receipts only after the target proxy, population, and metric terminology are settled.
