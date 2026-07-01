"""Reproducibly rebuild the precomputed rules artifact the Streamlit app loads.

Run from the project root:  python src/build_rules.py

Pipeline: raw -> clean -> basket matrix -> frequent itemsets (FP-Growth) -> rules.
The support / confidence floors here define the app's minimum slider values, so the
sliders only ever filter UP and never need to re-mine.
"""

from __future__ import annotations

from preprocess import build_basket_matrix, clean, load_raw
from mining import frequent_itemsets, generate_rules, rules_to_table

MIN_SUPPORT = 0.01      # itemset must appear in >= 1% of baskets
MIN_CONFIDENCE = 0.10   # rule must be right >= 10% of the time
OUT = "data/rules.csv"


def main() -> None:
    basket = build_basket_matrix(clean(load_raw())[0])
    print(f"basket matrix: {basket.shape[0]:,} baskets x {basket.shape[1]:,} products")

    itemsets, secs = frequent_itemsets(basket, "fpgrowth", min_support=MIN_SUPPORT)
    print(f"frequent itemsets (support >= {MIN_SUPPORT}): {len(itemsets):,}  ({secs:.2f}s)")

    rules = generate_rules(itemsets, min_confidence=MIN_CONFIDENCE, sort_by="lift")
    table = rules_to_table(rules)
    table.to_csv(OUT, index=False)
    print(f"rules (confidence >= {MIN_CONFIDENCE}): {len(table):,}  ->  wrote {OUT}")


if __name__ == "__main__":
    main()
