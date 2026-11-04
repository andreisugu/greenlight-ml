# Metrics and Financial Proxies

## Classification metrics

The primary population contains 3,229 rows with positive finite reported budget and revenue. On it, the nominal gross-to-budget label `revenue > budget` has prevalence 75.50%; the alternative `revenue >= 2 * budget` label has prevalence 56.02%.

These are selected-sample label prevalences, not proof of studio profitability. V1 reports a majority-class baseline, accuracy, balanced accuracy, precision, recall, macro F1, ROC-AUC, PR-AUC, and a confusion matrix on the same 646-title holdout. The full table is in [`V1_REPORT.md`](./V1_REPORT.md).

V1 result in one line: ROC-AUC 0.604 and PR-AUC 0.809 for the tuned linear model, versus 0.500 / 0.755 for the dummy. At 0.50 the model selects almost every title and stays at majority-class accuracy; the OOF-tuned 0.74 threshold selects 374/646 and reaches balanced accuracy 0.582 and macro F1 0.550.

## Financial proxy (not ROI)

The current arithmetic

```text
(sum(revenue) - sum(budget)) / sum(budget)
```

is named **Gross-to-Budget Surplus Proxy**. It is a gross box-office multiple less one, weighted by production budget. It is not net ROI or studio cash return: it does not account for exhibitor shares, Print & Advertising, distribution, financing, or other costs. The 2× label is likewise a rough target proxy, not an accounting result.

V1 does not publish a portfolio or ROI number. The gross-to-budget proxy is computed by `src/metrics.py` but is not used as V1 evidence.

## Status of historical benchmark values

Previously published benchmark receipts (including the 641-title holdout and ROI uplift claims) are withdrawn as validated evidence. They came from a pipeline that fit feature vocabularies/category choices before the train/test split, and they used a disputed >$10K population restriction. The leakage bug is fixed (`FeatureEncoder` + split-before-fit), and V1 now has a frozen contract and a published receipt. The remaining 03–04 notebooks are still drafts.
