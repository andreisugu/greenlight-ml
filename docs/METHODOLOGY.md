# 🔬 Methodology & Empirical Modeling

This document formalizes the analytical framework, experimental setup, and risk auditing protocols powering GreenlightML.

---

## 1. The Pre-Release Mandate

The fundamental business objective of film risk modeling is to support **greenlighting**—the executive decision to commit millions of dollars into production, talent contracts, and marketing. 

To be operationally useful, **all predictive signals must be knowable strictly prior to theatrical premiere**. Models trained on post-release attributes (audience ratings, post-release buzz) offer zero decision utility to a studio executive deciding whether to fund a screenplay 18–24 months before release.

---

## 2. The Survivorship Trap & Class Imbalance

Public film datasets inherently suffer from severe **survivorship bias**:
* Major studio films with extensive theatrical rollouts reliably report global box office receipts.
* Critical failures, direct-to-video releases, and unreleased indie productions frequently lack box office tracking and enter databases with `revenue == 0`.

When incomplete records (`revenue == 0 | budget == 0`) are filtered, the remaining 3,229 titles skew heavily toward commercial success:
* **Nominal Success ($Revenue > Budget$):** **2,438 titles (75.50%)**
* **Nominal Flop ($Revenue \le Budget$):** **791 titles (24.50%)**

### The Metric Trap
Under a 75.5% / 24.5% class distribution, a naive **Dummy Majority Classifier** predicting positive for 100% of cases achieves:
* **Accuracy:** `75.54%`
* **Precision (Class 1):** `75.54%`
* **Recall (Class 1):** `100.00%`
* **F1 Score (Class 1):** `0.8607`
* **ROC-AUC:** `0.5000`

> **Methodological Rule:** Any study reporting 75–78% raw accuracy or positive F1 on TMDB 5000 without reporting **Balanced Accuracy**, **Macro F1**, or **ROC-AUC** is demonstrating zero algorithmic lift beyond the majority class bias.

---

## 3. The Hollywood Break-Even Rule ($2.0\times$ Multiplier)

In entertainment economics, the simple condition $\text{Revenue} > \text{Budget}$ does **not** indicate a profitable theatrical investment:

1. **Exhibitor Box Office Split:** Gross box office represents total ticket sales. Cinema operators (exhibitors) retain ~50% of domestic gross (US/Canada), ~60% in Europe, and up to ~75% in China. The distributor/studio collects only the studio theatrical rental (~50% globally on average).
2. **Print & Advertising (P&A):** Production budgets cover only the physical cost of making the film. Global marketing, TV ad buys, junkets, and digital campaigns (P&A) routinely match 50% to 100% of the production budget.

### The Real-World Break-Even Benchmark:
$$\text{Worldwide Gross Revenue} \ge 2.0 \times \text{Production Budget}$$

When evaluated against the $2.0\times$ multiplier:
* **True Commercial Success:** **1,804 titles (55.87%)**
* **Theatrical Underperformance:** **1,425 titles (44.13%)**

The class distribution naturally balances to an approximate 56% / 44% ratio, providing a realistic, commercially grounded classification target.

---

## 4. Controlled Data Leakage Experiment

A pervasive defect in academic film modeling is the accidental inclusion of post-release feedback loops:
* `popularity`: TMDB's proprietary metric, recalculated daily based on current user page views, searches, and watchlist activity.
* `vote_average` & `vote_count`: Aggregated audience reviews gathered after viewers have seen the film.

### Quantitative Leakage Audit
In an 80/20 stratified experiment on 646 unseen test titles, injecting these three features produces an artificial lift:

| Model Architecture | Pre-Release Only (Clean) | Leaked (Post-Release) | Artificial Delta |
|---|:---:|:---:|:---:|
| **Logistic Regression ROC-AUC** | `0.6209` | `0.8139` | **+0.1930 (+31.1%)** |
| **Random Forest ROC-AUC** | `0.6101` | `0.8343` | **+0.2242 (+36.7%)** |
| **Logistic Regression Accuracy** | `74.30%` | `80.50%` | **+6.20%** |
| **Random Forest Accuracy** | `75.23%` | `81.11%` | **+5.88%** |

GreenlightML explicitly quarantines all post-release signals in production pipelines to ensure operational fidelity.

---

## 5. Validation Protocols

1. **Stratified 80/20 Holdout Split:** Preserves the 75.5% / 24.5% class balance in both training and test sets.
2. **Cross-Validation on Train Only:** Stratified 5-Fold Cross-Validation is executed strictly on the 80% training slice during `GridSearchCV` hyperparameter tuning to prevent test set contamination.
3. **Temporal Split (Optional / Rolling Window):** Splitting by `release_date` (e.g., training on $\le 2010$, testing on $2011–2016$) to audit against temporal leakage of director/actor fame.
