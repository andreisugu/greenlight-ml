# V1 Report — Regularized Linear Baseline

> Companion to [`notebooks/02_baseline_linear_model.ipynb`](../notebooks/02_baseline_linear_model.ipynb).
> The split receipt is committed as [`artifacts/split_contract.json`](../artifacts/split_contract.json).
> Every metric below is produced by the executed notebook.

## Contract v1.0

| Element | Frozen choice |
|---|---|
| Primary target | `revenue > budget` (`target_nominal`, 75.50%) |
| Sensitivity target | `revenue >= 2 × budget` (`target_hollywood`, 56.02%) |
| Population | 3,229 titles with positive finite reported budget and revenue |
| Prediction time | Pre-release only; `popularity`, `vote_average`, `vote_count` excluded |
| Split | Random 80/20 stratified, seed 42: 2,583 train / 646 test |
| Features | 29 basic pre-release features: log budget, runtime, year, month, 19 genres, top-5 languages + Other |
| Threshold | 0.50 reported, plus an out-of-fold tuned threshold |

V1 deliberately excludes synopsis TF-IDF and director/cast encoding; those belong to V2.

## Tuning

The pipeline is `FeatureEncoder → StandardScaler → LogisticRegression`, so the encoder and scaler are refit inside every fold. `GridSearchCV` uses `StratifiedKFold(5, shuffle=True, random_state=42)` and ROC-AUC.

- Grid: `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}`, `penalty ∈ {l1, l2}`, `class_weight ∈ {None, balanced}` (24 configurations).
- Best CV ROC-AUC: **0.6021** — `C=0.1`, `penalty=l1`, `class_weight=None`.
- OOF-tuned threshold: **0.74**, chosen for maximum balanced accuracy on training folds.

## Holdout results (646 titles)

| Model | Accuracy | Balanced Acc | Precision | Recall | Macro F1 | ROC-AUC | PR-AUC | Picked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (majority) | 0.7554 | 0.5000 | 0.7554 | 1.0000 | 0.4303 | 0.5000 | 0.7554 | 646/646 |
| Logistic Regression @ 0.50 | 0.7523 | 0.5044 | 0.7571 | 0.9898 | 0.4470 | 0.6042 | 0.8091 | 638/646 |
| Logistic Regression @ 0.74 | 0.6006 | 0.5816 | 0.8075 | 0.6189 | 0.5503 | 0.6042 | 0.8091 | 374/646 |

Confusion matrix at 0.74: **TN=86, FP=72, FN=186, TP=302**. At 0.50 the model selects almost everything, so its balanced accuracy stays near the dummy; the tuned threshold trades majority-class accuracy for a usable balance between precision and recall.

## Reading

The baseline carries real but weak signal: ROC-AUC 0.604 and PR-AUC 0.809 versus 0.500 / 0.755 for the dummy. Accuracy alone would hide this because the positive class is 75.5% of the holdout.

Limits: the random split mixes eras and recurring directors/actors, and the population is selected by financial reporting. This is a 1916–2016 reported-financials holdout, not a future-film or studio-profit claim. No portfolio/ROI number is published from V1.

## Reproduce

```bash
python -m unittest discover -s tests -v
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_linear_model.ipynb
```

The V1 environment pins `scikit-learn<1.8` because 1.8 deprecates the course-required `penalty` API.
