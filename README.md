# 🛒 Market Basket Analysis — Association Rule Mining

Mining **actionable co-purchase rules** from ~500k real e-commerce transactions
(UCI *Online Retail*), with an interactive Streamlit app to explore them.

> **This is unsupervised association-rule mining.** There is **no train/test split,
> no model training, and no accuracy / precision / recall / F1** — those metrics don't
> exist for pattern mining. The output is a ranked set of rules `{A} → {B}` judged by
> **support, confidence, and lift**, meant to inform business decisions
> (bundling, cross-sell, placement, inventory).

<!-- TODO: add a demo GIF of the Streamlit app here, e.g. ![demo](docs/demo.gif) -->
<!-- TODO: add the live app link after deploying, e.g. **Live app:** https://<your-app>.streamlit.app -->

---

## The business question

> *"What products are bought together, and what should we do about it?"*

Every rule maps to a concrete merchandising action:

| Lever | Example |
|---|---|
| **Bundling** | Sell matched sets together ("complete the collection") |
| **Cross-sell** | "Customers who bought X also bought Y" recommendations |
| **Placement** | Co-locate associated items on shelf / on-page |
| **Inventory** | Co-manage stock for items that sell as a set |

The stakeholder is a **merchandising manager**; the deliverable is a **ranked, filtered
list of rules**, not a model file.

---

## The three metrics (for a rule `A → B`)

| Metric | Formula | Reads as |
|---|---|---|
| **Support** | `P(A ∩ B)` | How often the itemset occurs (its *reach*) |
| **Confidence** | `P(B \| A) = supp(A∩B) / supp(A)` | Reliability & **direction** of the rule |
| **Lift** | `confidence / supp(B)` | Strength vs chance: **>1** positive, **=1** independent, **<1** substitutes |
| *Leverage* | `supp(A∩B) − supp(A)·supp(B)` | Absolute business impact (frequency × strength) |

**Why all three?** Support alone finds only *common* itemsets (often trivial).
Confidence alone is **fooled by popular consequents** — a rule to a best-seller can score
high confidence while meaning nothing. **Lift fixes this** by dividing out the baseline
(`lift = confidence / support(B)`). That's the classic *"everyone buys bananas"* trap.

---

## Key results (reproducible)

- **Data:** 541,909 raw lines → **527,725 clean** (97.4% kept); 25,900 invoices → **19,773 baskets**; **3,986 products**.
- **Basket matrix:** 19,773 × 3,986 boolean, **99.34% sparse** (~80 MB).
- **Frequent itemsets:** 1,892 at `support ≥ 0.01` (FP-Growth, ~1.2 s).
- **Rules:** **3,288** at `confidence ≥ 0.1`; **every rule has lift ≥ 1.36** (all positive associations).

### Apriori vs FP-Growth (same matrix, identical results, different speed)

| min_support | # itemsets | Apriori | FP-Growth | speedup |
|---|---|---|---|---|
| 0.03 | 142 | 0.50 s | 0.22 s | 2.3× |
| 0.02 | 384 | 2.24 s | 0.41 s | 5.5× |
| 0.01 | 1,892 | 21.54 s | 1.18 s | **18.3×** |

Both algorithms return **identical** itemsets (verified). FP-Growth wins because it builds
a compact **FP-tree** once and mines it recursively — **no candidate generation** and far
fewer data scans — so the gap **widens as support drops**.

### Example rules

| rule | support | confidence | lift |
|---|---|---|---|
| `HERB MARKER PARSLEY, ROSEMARY → HERB MARKER THYME` | 0.010 | 0.94 | **78.8** |
| `GREEN REGENCY TEACUP → ROSES REGENCY TEACUP` | 0.039 | 0.76 | 14.1 |
| `PINK POLKADOT bag → RED RETROSPOT bag` | 0.042 | 0.68 | 6.4 |

**Business actions:** "complete the collection" bundles (herb markers, Regency teacups),
directional cross-sell (recommend by *confidence*, not lift), catalog adjacency for
high-leverage pairs, and coordinated inventory for matched sets.
*(These are **associations, not causation** — each is a hypothesis to A/B test.)*

---

## Project structure

```
market-basket-analysis/
├── data/
│   ├── PROVENANCE.md            # dataset source, columns, quirks
│   └── rules.csv                # precomputed rules (committed; the app loads this)
├── notebooks/
│   └── 01_eda_and_mining.ipynb  # narrative: EDA → cleaning → mining → rules
├── src/
│   ├── preprocess.py            # cleaning + LONG→WIDE basket matrix
│   ├── mining.py                # Apriori/FP-Growth + rule generation
│   └── build_rules.py           # reproducibly rebuild data/rules.csv
├── app/
│   └── streamlit_app.py         # interactive rule explorer
├── requirements.txt
└── README.md
```

Large raw files (`Online Retail.xlsx`, `*_raw.csv`, `*_clean.csv`) are **git-ignored** and
re-downloadable — see [`data/PROVENANCE.md`](data/PROVENANCE.md).

---

## Run it locally

```bash
# 1. install
pip install -r requirements.txt

# 2. get the raw data (~23 MB) into data/
curl -L -o "data/Online Retail.xlsx" \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"

# 3. (optional) rebuild the rules artifact from scratch
python src/build_rules.py        # writes data/rules.csv

# 4. launch the app
streamlit run app/streamlit_app.py
```

The app **loads the precomputed `rules.csv` once** (`st.cache_data`) and filters it
in-memory, so the support/confidence/lift sliders are instant — nothing is re-mined.

---

## Pipeline (how it was built)

1. **Problem framing** — define the decision the rules inform.
2. **Data acquisition** — load raw `Online Retail`, document columns/provenance.
3. **EDA** — basket sizes, popular/rare items, quantify junk.
4. **Cleaning** — drop returns, non-product codes (`POSTAGE`, `M`, …), bad prices, blanks.
5. **Basket matrix** — LONG → WIDE boolean one-hot (presence, not quantity), verified.
6. **Mining** — Apriori vs FP-Growth, runtime comparison.
7. **Rules** — support / confidence / lift, the "bananas" trap, substitutes (lift < 1).
8. **Interpretation** — top rules → plain-English business actions.

---

## Caveats (stated honestly)

- **Association ≠ causation.** Rules are hypotheses to test, not proven levers.
- **Modest reach.** Even top rules sit at 1–4% support — real signal, not universal.
- **Within-range dominance.** Associations are mostly "more variants of the same range"
  — genuine for giftware, but worth validating with a promotion test.
- **No predictive metric.** By design — this is descriptive pattern mining.

## Data source & citation

UCI Machine Learning Repository — *Online Retail* (ID 352).
Chen, D., Sain, S.L., & Guo, K. (2012). *Data mining for the online retail industry.*
Journal of Database Marketing & Customer Strategy Management, 19(3).
