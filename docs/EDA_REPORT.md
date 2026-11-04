# EDA Report — TMDB 5000

> Written companion to [`notebooks/01_exploratory_data_analysis.ipynb`](../notebooks/01_exploratory_data_analysis.ipynb).
> Every number in §1–§4 is printed or plotted by that notebook.

## 1. Merge and missingness

An inner join on `id = movie_id` gives **4,803 rows × 22 columns**. The credits file repeats the title, so the join keeps one
copy instead of carrying `title_credits`.

The file is unevenly populated:

| Column | Missing |
|---|---:|
| `homepage` | 3,091 |
| `tagline` | 844 |
| `overview` | 3 |
| `runtime` | 2 |
| `release_date` | 1 |
| everything else | 0 |

The sparse fields are marketing text, not measurements. The columns the analysis needs — budget, revenue, genres, crew,
language — are complete in this dump.

## 2. The reported-financials subset

A zero budget or revenue is treated as **unreported**, not as a value: the notebook keeps rows with `budget > 0` and
`revenue > 0`, which leaves **3,229 of 4,803** titles. The filter does not say *why* a value is absent.

| | budget (USD) | revenue (USD) | log1p(budget) | log1p(revenue) |
|---|---:|---:|---:|---:|
| count | 3,229 | 3,229 | 3,229 | 3,229 |
| mean | 40.7M | 121.2M | 16.80 | 17.49 |
| std | 44.4M | 186.3M | 1.66 | 2.08 |
| min | 1 | 5 | 0.69 | 1.79 |
| median | 25.0M | 55.2M | 17.03 | 17.83 |
| max | 380.0M | 2,787.9M | 19.76 | 21.75 |

Both columns span several orders of magnitude between median and maximum. `log1p` compresses that spread, which is why
the plots use it.

## 3. Relationships

The scatterplot of `log1p(budget)` against `log1p(revenue)` shows a positive but noisy relationship. Correlations on the
same subset:

| Pair | r |
|---|---:|
| `log_budget` — `log_revenue` | 0.66 |
| `log_revenue` — `popularity` | 0.42 |
| `log_revenue` — `vote_average` | 0.15 |
| `log_budget` — `vote_average` | −0.10 |

Budget and revenue move together, but far from perfectly: budget alone leaves most of the variance unexplained, which is
consistent with the weak signal the V1 baseline finds on this population.

`popularity` and `vote_average` are **post-release** fields. Their correlation with revenue is descriptive; it is not
evidence that they exist at a pre-release decision point, and they stay out of the V1 feature set.

## 4. Genres and directors

Genres are multi-label — one title can count under several — and the subset contains **19 labels**. The twelve most
frequent are:

| Genre | Titles |
|---|---:|
| Drama | 1,441 |
| Comedy | 1,110 |
| Thriller | 935 |
| Action | 918 |
| Adventure | 661 |
| Romance | 574 |
| Crime | 521 |
| Science Fiction | 431 |
| Family | 365 |
| Fantasy | 342 |
| Horror | 332 |
| Mystery | 265 |

The long tail matters for encoding: rare labels cannot support their own column on a 2,583-title training split, so the
feature encoder collapses everything outside a fitted vocabulary into `Other`.

Directors are extracted from the credits JSON. The subset has **1,445 distinct values**, one of which is an `Unknown`
bucket with 2 rows. The most frequent:

| Director | Titles |
|---|---:|
| Steven Spielberg | 27 |
| Clint Eastwood | 19 |
| Ridley Scott | 16 |
| Martin Scorsese | 16 |
| Robert Rodriguez | 16 |
| Renny Harlin | 15 |
| Tim Burton | 14 |
| Steven Soderbergh | 14 |
| Robert Zemeckis | 13 |
| Oliver Stone | 13 |
| Michael Bay | 12 |
| Joel Schumacher | 12 |

## 5. Limits

- **Selection by reporting.** The 3,229 rows are conditional on reported finances. Excluded films may differ
  systematically, so no rate computed here is an industry rate.
- **Proxies, not profit.** Budget and revenue are gross figures without exhibitor splits, marketing, distribution, or
  financing. `revenue > budget` is a gross-to-budget proxy, not studio profit.
- **No temporal control.** The subset spans 1916–2016 and this pass does not split it by era; the relationships above are
  pool-wide. Temporal drift is documented in [`DATA_DICTIONARY.md`](./DATA_DICTIONARY.md).
- **Rows are not independent.** Directors and lead actors recur across titles, which is why a plain random split has to be
  read as a holdout for this population only, not as evidence about future films.

## Reproduce

```bash
python -m unittest discover -s tests -v
jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploratory_data_analysis.ipynb
```
