# 💰 Metrics & Financial Portfolio Modeling

This document details the dual evaluation framework of GreenlightML: statistical classification metrics and the **Simulated Net Portfolio ROI** financial decision engine.

---

## 1. Classification Metrics (Statistical Evaluation)

Due to the 75.5% majority class imbalance, single-point accuracy is fundamentally misleading. GreenlightML evaluates models across a holistic metric battery:

* **Balanced Accuracy:** $\frac{1}{2} (\text{Sensitivity} + \text{Specificity}) = \frac{1}{2} (\text{TPR} + \text{TNR})$. Penalizes models that achieve high raw accuracy by ignoring the minority flop class.
* **Precision ($\text{PPV}$):** $\frac{\text{TP}}{\text{TP} + \text{FP}}$. Measures the percentage of predicted "greenlit" titles that actually broke even.
* **Recall ($\text{Sensitivity}$):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$. Measures the fraction of true box office hits captured by the model.
* **Macro F1 Score:** $\frac{\text{F1}_{\text{hit}} + \text{F1}_{\text{flop}}}{2}$. Provides unweighted average F1 across both classes, immediately revealing if a model is failing on the 24.5% flop class.
* **ROC-AUC:** Area under the Receiver Operating Characteristic curve. Measures ranking discrimination threshold-independently ($0.50$ = random chance, $1.00$ = perfect ranking).

---

## 2. Simulated Net Portfolio ROI (Financial Evaluation)

In financial entertainment analytics, classification accuracy does not answer a venture capitalist's or studio head's primary question: **"Does this model generate superior capital returns compared to indexing the theatrical market?"**

### The Capital Allocation Metric
For any classification decision threshold where titles with predicted probability $\hat{p}_i \ge \tau$ are selected ("greenlit"):

$$\text{Net Portfolio ROI} = \frac{\sum_{i \in \text{Selected}} (\text{Revenue}_i - \text{Budget}_i)}{\sum_{i \in \text{Selected}} \text{Budget}_i}$$

### Why Unweighted Mean ROI Fails
Calculating the simple arithmetic mean of individual movie ROIs:
$$\text{Mean ROI}_{\text{naive}} = \frac{1}{N} \sum_{i=1}^N \frac{\text{Revenue}_i - \text{Budget}_i}{\text{Budget}_i}$$
is catastrophic on real film data because ultra-low-budget viral indies completely distort the mean:
* *Tarnation* (2003): Budget \$218 $\rightarrow$ Revenue \$1.16M $\rightarrow$ Individual ROI: **+532,933%**.
* *Paranormal Activity* (2007): Budget \$15,000 $\rightarrow$ Revenue \$193.3M $\rightarrow$ Individual ROI: **+1,288,938%**.

GreenlightML's **weighted portfolio aggregation** pools all capital and revenues at the fund level, accurately reflecting how an institutional film investment vehicle operates.

---

## 3. Empirical Holdout Performance

Across 646 unseen theatrical test titles:

```text
Total Test Theatrical Market Capital:    $26.35 Billion
Total Test Theatrical Market Gross:      $73.18 Billion
Passive Theatrical Market Net Profit:    $46.83 Billion
Passive Market Index ROI:                +177.76% (Fund all 646 titles)
```

### Model-Driven Allocation Performance

| Strategy | Titles Picked | Capital Deployed | Capital Saved | Net Portfolio ROI | ROI Delta vs. Market |
|---|:---:|:---:|:---:|:---:|:---:|
| **Passive Market Index** | 646 | \$26.35B | \$0.00B | `+177.76%` | Baseline |
| **Logistic Regression (Pre-release)** | **606** | **\$24.69B** | **\$1.66B** | **`+181.76%`** | **+4.00% Alpha** |
| **Random Forest (Pre-release)** | 624 | \$25.68B | \$0.67B | `+179.65%` | +1.89% Alpha |
| **Oracle (Perfect Foresight)** | 488 | \$19.82B | \$6.53B | `+237.28%` | +59.52% Upper Bound |

### Strategic Takeaway
Even though creative variance limits pre-release accuracy to ~75%, **Logistic Regression successfully rejects 40 high-probability commercial flops**, preserving \$1.66 Billion in capital and lifting net fund returns by **+400 basis points (+181.76% vs +177.76%)**.
