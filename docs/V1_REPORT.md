# V1 — a first baseline model

> Companion to [`notebooks/02_baseline_linear_model.ipynb`](../notebooks/02_baseline_linear_model.ipynb).
> The split receipt is in [`artifacts/split_receipt.json`](../artifacts/split_receipt.json).
> Every number below is printed by the executed notebook.

**Short version:** a simple model ranks a hit above a miss about 60% of the time. That is a little signal, not a lot;
at the default cut-off it does no better than always answering "hit".

## The question

Can a simple model tell a film that earned more than its budget from one that did not, using only information that
exists before release — and does it beat always answering "it will earn more"?

## Settings

| Setting | Value |
|---|---|
| "Hit" means | `revenue > budget` — box-office gross above production budget; 2,438 of 3,229 titles (75.50%) |
| Films used | 3,229 with a positive reported budget and revenue |
| Train / test | 80% / 20% at random, keeping the 75.5% hit rate in both halves; seed 42 → 2,583 / 646 |
| Inputs | 29 columns recorded before release: budget on a log scale, runtime, year, month, 19 genres, 5 languages + Other |
| Threshold | A 0.50 cut-off, plus one chosen from the training rows only (0.74) |

`popularity`, `vote_average` and `vote_count` are left out because they are recorded after release. Synopsis text and
director/cast features are V2.

Whether each of the 29 columns actually existed at a fixed decision date is **assumed, not audited**; that check is
still open.

## How it is trained

The model is **logistic regression**: it fits a straight boundary through the inputs and turns the result into a
probability between 0 and 1. It is the simplest model that can use 29 features, and its coefficients can be read.

Preprocessing (`FeatureEncoder`) and scaling (`StandardScaler`) sit inside a scikit-learn `Pipeline`. A pipeline is just
a fixed sequence of steps; putting it inside the tuning loop means the preprocessing is fit on the four training folds
and applied to the fifth, never on the test rows.

**Cross-validation** splits the 2,583 training rows into 5 parts, scores each part with a model trained on the other
four, and averages. The 24 hyperparameter combinations (settings the model does not learn from the data) are
compared on that average:

- `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` — how hard large coefficients are penalised
- `penalty ∈ {l1, l2}` — which kind of penalty: l1 can push useless coefficients to zero, l2 only shrinks them
- `class_weight ∈ {None, balanced}` — whether the two classes are weighted equally or by frequency

Best cross-validated ROC-AUC: **0.6021** (`C=0.1`, `l1`, `None`).

## How the cut-off is chosen

The model outputs a probability, so a cut-off is needed to turn it into a decision.

Every training row is scored by the fold-model that did not see it. Those are **out-of-fold (OOF)** predictions. The
cut-off that gives the best balanced accuracy on them is **0.74**. The test rows play no part in this choice, so the
holdout stays clean.

Those folds also chose the hyperparameters, which makes the cut-off mildly optimistic. That affects the choice of
cut-off, not the holdout score reported below.

## Results on the 646 test rows

| Model | Accuracy | Balanced Acc | Precision | Recall | Macro F1 | ROC-AUC | PR-AUC | Picked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always "hit" | 0.7554 | 0.5000 | 0.7554 | 1.0000 | 0.4303 | 0.5000 | 0.7554 | 646/646 |
| Regression @ 0.50 | 0.7523 | 0.5044 | 0.7571 | 0.9898 | 0.4470 | 0.6042 | 0.8091 | 638/646 |
| Regression @ 0.74 | 0.6006 | 0.5816 | 0.8075 | 0.6189 | 0.5503 | 0.6042 | 0.8091 | 374/646 |

At the 0.74 cut-off: **86** correctly rejected, **72** false alarms, **186** missed hits, **302** correctly found.

## What each number means

| Term | Plain meaning | "No skill" value |
|---|---|---|
| Accuracy | Share of titles labelled correctly | 0.755 (always "hit") |
| Balanced accuracy | Hit-recall and miss-recall, averaged | 0.500 |
| Precision | Of the titles called hits, the share that are | 0.755 |
| Recall | Of the real hits, the share found | 1.000 (always "hit") |
| Macro F1 | Precision and recall combined, both classes weighted equally | 0.430 |
| ROC-AUC | Chance a random hit outranks a random miss | 0.500 |
| PR-AUC | Same idea, focused on the titles selected | 0.755 |
| Threshold | Probability above which a title is called a hit | — |
| Cross-validation | Score each of 5 parts with a model trained on the other 4 | — |
| OOF | A training row scored by a model that did not see it | — |

**In one sentence:** the model ranks a hit above a miss about 60% of the time (0.604, against 0.500 for a coin flip).
That is a little signal, not a lot. At the 0.50 cut-off it behaves like the baseline; at 0.74 it finds about 619 of
every 1,000 hits, and about 808 of every 1,000 titles it picks are really hits.

## What this does not say

- It does not predict **future** films. The split is random, so it mixes 1916–2016 and lets the same directors and
  actors appear on both sides.
- It does not describe the industry. The sample is limited to films that reported their finances.
- It does not establish that the inputs existed at a fixed decision date; that audit is still open.
- It publishes no portfolio or ROI number.

## Reproduce

```bash
python -m unittest discover -s tests -v
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_linear_model.ipynb
```

The environment pins `scikit-learn<1.8` because 1.8 deprecates the `penalty` parameter the grid uses; removal is
scheduled for 1.10.
