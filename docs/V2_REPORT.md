# V2 — text, people, trees

> Companion to [`notebooks/03_advanced_ensemble_model.ipynb`](../notebooks/03_advanced_ensemble_model.ipynb).
> Every number below is printed by the executed notebook.

**Short version:** adding the synopsis, the director and the lead actors raises the ranking score from 0.604 to
**0.709**. Most of that comes from how often a director and the lead actor appear in the training data, not from the
story text.

## The question

V1 used 29 structured columns. Does a wider feature set — free text plus the people attached to a film — make the
model better at telling a hit from a miss, on exactly the same rows?

## Settings

| Setting | Value |
|---|---|
| "Hit" means | `revenue > budget` — 2,438 of 3,229 titles (75.50%) |
| Films, split | the same as V1: 3,229 rows, 2,583 / 646, seed 42 |
| V1 features | 29: log budget, runtime, year, month, genres, languages |
| V2 features | 215: those 29, plus 31 director columns, 5 people-frequency columns, 150 synopsis terms |
| Text | lowercased, digits and punctuation removed, English stop words dropped, 1–2 word terms |
| Models | XGBoost (120 random draws) and Random Forest (36 combinations) |
| Cut-off | 0.50 for the comparison; 0.79 chosen from out-of-fold training predictions |

Population, target and split are identical to V1. Only the inputs and the model change, which is what makes the
comparison mean anything.

## How it is trained

**Text.** Each synopsis becomes 150 TF-IDF columns. A term scores high when it is frequent in one film and rare across
films; unigrams and bigrams share the 150 slots, and in practice 4 bigrams survive (`new york`, `world war`, …).

**People.** Each name is replaced by its share of the training rows. An actor who never appears in training scores 0,
so the test rows cannot introduce new vocabulary. This is frequency encoding rather than target encoding: it never
looks at the outcome, so it cannot leak the label.

**Models.** XGBoost adds trees in sequence, each correcting the previous one; the search covers `n_estimators`,
`max_depth`, `learning_rate`, `subsample`, `colsample_bytree` and `scale_pos_weight`. Random Forest averages independent
trees and serves as a second opinion. Both are scored by 5-fold cross-validation on the 2,583 training rows, and both
are re-fitted inside every fold.

## Results on the 646 test rows, cut-off 0.50

| Model | Accuracy | Balanced Acc | Precision | Recall | Macro F1 | ROC-AUC | PR-AUC | Picked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always "hit" | 0.7554 | 0.5000 | 0.7554 | 1.0000 | 0.4303 | 0.5000 | 0.7554 | 646/646 |
| V1 logistic regression | 0.7523 | 0.5044 | 0.7571 | 0.9898 | 0.4470 | 0.6042 | 0.8091 | 638/646 |
| **V2 XGBoost** | 0.7601 | 0.5309 | 0.7673 | 0.9795 | 0.5021 | **0.7093** | **0.8741** | 623/646 |
| V2 Random Forest | 0.7570 | 0.5032 | 0.7566 | 1.0000 | 0.4370 | 0.6779 | 0.8596 | 645/646 |

Best cross-validated ROC-AUC: **0.6911** for XGBoost (300 trees, depth 10, learning rate 0.01, subsample 0.8,
`colsample_bytree` 0.7, `scale_pos_weight` 1.0) and 0.6679 for Random Forest.

At the 0.79 cut-off XGBoost finds **46%** of hits with **89%** precision and selects 251 of 646 titles, against 98% recall
and 77% precision at 0.50. Which is preferable depends on the cost of the two errors, not on the model.

## New terms

| Term | Plain meaning |
|---|---|
| TF-IDF | Word score = frequent in this film, rare overall |
| Bigram | Two adjacent words treated as one term |
| Frequency encoding | Replace a name with its share of training rows (unseen → 0) |
| Gain | XGBoost's own count of how much a column improved its trees |
| Permutation importance | Shuffle one column, measure the drop in ROC-AUC |
| `scale_pos_weight` | Weight on the positive class; 1.0 treats both alike, 0.33 leans against "hit" |

The two importance views disagree, and that is informative. Gain is spread across many TF-IDF columns because the trees
use them often. Permutation puts `director_freq` (0.057), `release_year` (0.052), `log_budget` (0.041) and `cast1_freq`
(0.017) far ahead: no single synopsis word carries much, because the other 149 cover for it.

## What this does not say

- **It does not prove text is useless.** Text contributes, but diffusely. The measurable jump comes from people
  frequencies, which is a weaker and more suspicious feature than it looks: it encodes *who was cast*, and casting
  correlates with budget and era.
- **The split flatters V2 specifically.** The same directors and actors appear on both sides of a random split, so a
  frequency feature can partially memorise the training rows. A group-aware split is the honest test and is not in V2.
- **It is not a claim about future films**, nor about the industry: the population is limited to films that reported
  their finances, and whether these columns existed at a fixed decision date is assumed, not audited.
- **No portfolio or ROI number is published.**

## Reproduce

```bash
python -m unittest discover -s tests -v
jupyter nbconvert --to notebook --execute --inplace notebooks/03_advanced_ensemble_model.ipynb
```
