# 📚 Data Dictionary & Schema Audit

This document outlines the relational schema, feature definitions, and data quality audit for the **TMDB 5000 Movie Dataset** used in GreenlightML.

---

## 1. Source & File Structure

The dataset originates from [The Movie Database (TMDB)](https://www.themoviedb.org) aggregated metadata and comprises two core CSV files. This is a later snapshot; a field's apparent pre-release meaning does not prove that its value was available at a historical prediction date. Treat temporal labels below as hypotheses until provenance is checked.

* **`tmdb_5000_movies.csv`**: 4,803 rows × 20 columns. Contains financial, content, production, and reception attributes.
* **`tmdb_5000_credits.csv`**: 4,803 rows × 4 columns. Contains JSON-encoded cast billing and production crew metadata.

### The 22 Unique Features vs. 24 Raw Columns

While a simple sum of columns yields $20 + 4 = 24$, an inner join on `id = movie_id` reveals that there are **only 22 unique features**:
1. `movie_id` in `credits.csv` is a redundant foreign key identical to `id` in `movies.csv`.
2. `title` in `credits.csv` is 100% duplicate character-by-character with `title` in `movies.csv` across all 4,803 rows.

---

## 2. Complete Feature Taxonomy

| Column Name | Raw Data Type | Temporal Scope | Description & Modeling Role | Missing Values (Raw) |
|---|:---:|:---:|---|:---:|
| `id` | Integer | Metadata | Unique TMDB film identifier. Used solely as the relational join key. | 0 |
| `title` | String | Pre-release | International commercial release title. | 0 |
| `original_title` | String | Pre-release | Original language title (differs from `title` in 5.43% of cases). | 0 |
| `budget` | Integer | **Pre-release** | Stated production budget in USD. Zero indicates missing data. Transform to be chosen empirically (log not assumed). | 1,037 (zeros) |
| `revenue` | Integer | **Post-release** | Worldwide theatrical gross box office in USD. Used only to derive the target proxy (not a feature). | 1,427 (zeros) |
| `genres` | JSON String | **Pre-release** | List of genre dicts (`id`, `name`). Multi-label encoded; encoder must be fit on training data only. | 28 empty lists |
| `runtime` | Float | **Pre-release** | Film duration in minutes. Median imputation must be fit on training data only. | 2 NaN, 35 zeros |
| `release_date` | Date String | **Pre-release** | Theatrical premiere date (`YYYY-MM-DD`, 1916–2016). Used for temporal features & split. | 1 NaN |
| `original_language` | String | **Pre-release** | Primary dialogue language code (`en`, `fr`, `es`, etc.). | 0 |
| `spoken_languages` | JSON String | **Pre-release** | List of languages spoken in the film. | 86 empty lists |
| `production_countries`| JSON String | **Pre-release** | List of countries involved in production. | 174 empty lists |
| `production_companies`| JSON String | **Pre-release** | Studios and production entities (Warner Bros, Universal, etc.). | 351 empty lists |
| `status` | Categoric | Production | Lifecycle state: 4,795 Released, 5 Rumored, 3 Post Production. | 0 |
| `keywords` | JSON String | Pre-release* | Plot and thematic keyword tags. Quarantined if post-release tropes exist. | 412 empty lists |
| `overview` | Text String | **Pre-release** | Narrative plot synopsis. Vectorized via TF-IDF (100–150 unigrams/bigrams). | 3 NaN |
| `tagline` | Text String | Pre-release | Promotional marketing slogan. | 844 NaN |
| `homepage` | URL String | Pre-release | Official promotional web presence. Binarized into `has_homepage`. | 3,091 NaN |
| `cast` | JSON String | **Pre-release** | Cast billing order (`name`, `character`, `order`). Top 3 lead billing extracted. | 43 empty lists |
| `crew` | JSON String | **Pre-release** | Full production crew. Extracted primary director (`job == 'Director'`). | 28 empty lists |
| `popularity` | Float | **POST-RELEASE ⚠️**| Dynamic TMDB traffic, search, and watchlist index. **Data leakage signal.** | 1 zero |
| `vote_average` | Float | **POST-RELEASE ⚠️**| User rating average (0.0–10.0 scale). **Data leakage signal.** | 63 zeros |
| `vote_count` | Integer | **POST-RELEASE ⚠️**| Cumulative volume of user ratings. **Data leakage signal.** | 62 zeros |

---

## 3. Data Cleaning & Sanity Audit

### Pruning Missing Financials
* Unreported budgets: **1,037 films (21.59%)**
* Unreported revenues: **1,427 films (29.71%)**
* Logical OR (`budget == 0 | revenue == 0`): **1,574 films (32.77%)**
* **Reported-financials population:** **3,229 rows** remain after excluding zero budget/revenue sentinels. This is not proof every row was theatrically released.

### $10K scale threshold: sensitivity only

A strict `budget > $10,000 AND revenue > $10,000` restriction leaves 3,203 rows (26 fewer than the primary reported-financials population). It is not an anomaly detector: the removed rows include both revenue-above-budget and revenue-at/below-budget cases, and the CSV alone cannot establish that all their values are corrupt. Examples such as *Primer*, *Tarnation*, and *Eraserhead* require source-level verification before exclusion or correction. See the [EDA notebook](../notebooks/01_exploratory_data_analysis.ipynb) for the full list and label-sensitivity receipts.

Zero values are treated as unavailable financial reports in this analysis. They are not imputed to zero profit and are not automatically interpreted as commercial failures.
