"""Frequent itemset mining for market basket analysis.

Two algorithms, IDENTICAL output, DIFFERENT speed:

* Apriori   - breadth-first. GENERATES candidate itemsets level by level (1-itemsets,
              then 2-itemsets, ...) and rescans the data to count each candidate.
              Candidate generation is the bottleneck: candidates explode
              combinatorially as min_support drops.
* FP-Growth - builds a compact prefix tree (the FP-tree) of all transactions in ONE
              pass, then mines patterns recursively straight from the tree. NO
              candidate generation and far fewer data scans -> usually much faster
              at the same min_support.

Both return every itemset whose support >= min_support, so their RESULTS must be
identical; only RUNTIME differs. ``compare()`` proves that.

``min_support`` is a FRACTION of baskets (0.02 = "appears in >= 2% of baskets").
"""

from __future__ import annotations

import time

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

_ALGORITHMS = {"apriori": apriori, "fpgrowth": fpgrowth}

# Separator between members of an itemset in the serialized CSV. Must NOT appear in
# any product Description -- 68 descriptions contain commas (e.g. "TRAY, BREAKFAST IN
# BED"), but none contain "|", so a comma separator would be ambiguous and this is safe.
ITEM_SEP = " | "

# Columns we keep for the readable/serialized rules table (drop mlxtend's exotic ones).
_RULE_COLS = [
    "antecedents",
    "consequents",
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
]


def frequent_itemsets(
    basket: pd.DataFrame,
    algorithm: str = "fpgrowth",
    min_support: float = 0.02,
    max_len: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Run one algorithm and return (itemsets_df, runtime_seconds).

    ``use_colnames=True`` makes itemsets show product names instead of column
    indices -- essential for readable rules downstream.
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"algorithm must be one of {list(_ALGORITHMS)}")
    func = _ALGORITHMS[algorithm]
    t0 = time.perf_counter()
    itemsets = func(basket, min_support=min_support, use_colnames=True, max_len=max_len)
    return itemsets, time.perf_counter() - t0


def _canonical(itemsets: pd.DataFrame) -> dict[frozenset, float]:
    """{frozenset(items): rounded support} -- an order-independent comparison key."""
    return {frozenset(r.itemsets): round(r.support, 10) for r in itemsets.itertuples()}


def compare(
    basket: pd.DataFrame,
    min_support: float = 0.02,
    max_len: int | None = None,
) -> dict[str, tuple[pd.DataFrame, float]]:
    """Run BOTH algorithms, print runtimes, and assert their results are identical."""
    results: dict[str, tuple[pd.DataFrame, float]] = {}
    for name in ("apriori", "fpgrowth"):
        itemsets, secs = frequent_itemsets(basket, name, min_support, max_len)
        results[name] = (itemsets, secs)
        print(f"  {name:9s}: {len(itemsets):>6,} itemsets in {secs:7.3f}s")

    identical = _canonical(results["apriori"][0]) == _canonical(results["fpgrowth"][0])
    speedup = results["apriori"][1] / results["fpgrowth"][1]
    print(f"  identical itemsets & supports: {identical}")
    print(f"  FP-Growth speedup: {speedup:.1f}x")
    return results


def generate_rules(
    itemsets: pd.DataFrame,
    min_confidence: float = 0.1,
    sort_by: str = "lift",
) -> pd.DataFrame:
    """Turn frequent itemsets into association rules, sorted by ``sort_by``.

    We gate on CONFIDENCE (not lift) so the pool still contains lift < 1 rules
    (substitutes) for exploration; the Streamlit sliders filter from here. The
    ``min_confidence`` floor becomes the app's minimum-confidence slider value.

    Metrics (for rule  A -> B, "if a basket has A it also has B"):
      support    = P(A and B)              -- how often the pair co-occurs overall
      confidence = P(B | A) = supp(AB)/supp(A)  -- reliability of the rule
      lift       = confidence / supp(B) = P(AB)/(P(A)P(B))
                   > 1 positive assoc, = 1 independent, < 1 substitutes
      leverage   = supp(AB) - supp(A)supp(B)    -- co-occurrence above independence
      conviction = (1 - supp(B)) / (1 - confidence)  -- how "wrong" A->B would be if independent
    """
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    return rules.sort_values(sort_by, ascending=False).reset_index(drop=True)


def rules_to_table(rules: pd.DataFrame) -> pd.DataFrame:
    """Serialisable view: frozenset itemsets -> sorted comma-joined strings."""
    out = rules.copy()
    out["antecedents"] = out["antecedents"].apply(lambda s: ITEM_SEP.join(sorted(s)))
    out["consequents"] = out["consequents"].apply(lambda s: ITEM_SEP.join(sorted(s)))
    return out[_RULE_COLS]


if __name__ == "__main__":
    from preprocess import load_raw, clean, build_basket_matrix

    basket = build_basket_matrix(clean(load_raw())[0])
    print(f"basket matrix: {basket.shape[0]:,} baskets x {basket.shape[1]:,} products\n")
    for ms in (0.03, 0.02, 0.01):
        n = int(ms * basket.shape[0])
        print(f"min_support = {ms}  (>= {n:,} baskets)")
        compare(basket, min_support=ms)
        print()
