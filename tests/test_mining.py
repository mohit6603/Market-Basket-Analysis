"""Tests for frequent-itemset mining and rule generation.

Two layers:
1. Hand-computed values on the synthetic baskets (supports, confidence, lift).
2. Metric INVARIANTS that must hold for any rule set, checked over all rules
   (lift == confidence / consequent-support, support bounds, etc.).
"""

import pandas as pd
import pytest

from mining import ITEM_SEP, frequent_itemsets, generate_rules, rules_to_table
from preprocess import build_basket_matrix, clean


@pytest.fixture(scope="module")
def basket(raw_synthetic):
    cleaned, _ = clean(raw_synthetic)
    return build_basket_matrix(cleaned)


def _support_map(itemsets: pd.DataFrame) -> dict[frozenset, float]:
    return {frozenset(r.itemsets): r.support for r in itemsets.itertuples()}


def test_apriori_and_fpgrowth_return_identical_itemsets(basket):
    its_a, _ = frequent_itemsets(basket, "apriori", min_support=0.2)
    its_f, _ = frequent_itemsets(basket, "fpgrowth", min_support=0.2)
    a, f = _support_map(its_a), _support_map(its_f)
    assert a.keys() == f.keys()
    for k in a:
        assert a[k] == pytest.approx(f[k])


def test_supports_match_hand_computation(basket):
    its, _ = frequent_itemsets(basket, "fpgrowth", min_support=0.2)
    sup = _support_map(its)
    # single items (out of 5 baskets)
    assert sup[frozenset({"BREAD"})] == pytest.approx(4 / 5)
    assert sup[frozenset({"BUTTER"})] == pytest.approx(4 / 5)
    assert sup[frozenset({"MILK"})] == pytest.approx(2 / 5)
    # pairs
    assert sup[frozenset({"BREAD", "BUTTER"})] == pytest.approx(3 / 5)
    assert sup[frozenset({"BREAD", "MILK"})] == pytest.approx(2 / 5)
    assert sup[frozenset({"BUTTER", "MILK"})] == pytest.approx(1 / 5)


def test_min_support_floor_is_respected(basket):
    its, _ = frequent_itemsets(basket, "fpgrowth", min_support=0.5)
    sup = _support_map(its)
    assert all(s >= 0.5 for s in sup.values())
    # MILK (0.4) and every pair except BREAD+BUTTER must be gone
    assert frozenset({"MILK"}) not in sup
    assert frozenset({"BREAD", "BUTTER"}) in sup


def test_rule_metrics_match_hand_computation(basket):
    its, _ = frequent_itemsets(basket, "fpgrowth", min_support=0.2)
    rules = generate_rules(its, min_confidence=0.1)

    def rule(a, c):
        m = rules[
            (rules["antecedents"] == frozenset(a)) & (rules["consequents"] == frozenset(c))
        ]
        assert len(m) == 1, f"rule {a}->{c} missing"
        return m.iloc[0]

    r = rule({"BREAD"}, {"BUTTER"})
    assert r["confidence"] == pytest.approx(0.75)
    assert r["lift"] == pytest.approx(0.9375)  # < 1: mild substitutes by design

    r = rule({"BREAD"}, {"MILK"})
    assert r["confidence"] == pytest.approx(0.50)
    assert r["lift"] == pytest.approx(1.25)


def test_rule_metric_invariants_hold_for_every_rule(basket):
    its, _ = frequent_itemsets(basket, "fpgrowth", min_support=0.2)
    rules = generate_rules(its, min_confidence=0.1)
    assert len(rules) > 0
    # definitional identities, must hold row by row
    assert rules["confidence"].equals(rules["support"] / rules["antecedent support"]) or (
        (rules["confidence"] - rules["support"] / rules["antecedent support"]).abs() < 1e-12
    ).all()
    assert ((rules["lift"] - rules["confidence"] / rules["consequent support"]).abs() < 1e-12).all()
    # a pair can never be more frequent than either of its parts
    assert (rules["support"] <= rules["antecedent support"] + 1e-12).all()
    assert (rules["support"] <= rules["consequent support"] + 1e-12).all()
    # probabilities stay in [0, 1]
    for col in ("support", "confidence", "antecedent support", "consequent support"):
        assert rules[col].between(0, 1).all()


def test_rules_to_table_serializes_with_safe_separator(basket):
    its, _ = frequent_itemsets(basket, "fpgrowth", min_support=0.2)
    table = rules_to_table(generate_rules(its, min_confidence=0.1))
    assert table["antecedents"].map(lambda s: isinstance(s, str)).all()
    # multi-item itemsets must use ITEM_SEP (the comma-safe separator)
    multi = table[table["antecedents"].str.contains(ITEM_SEP, regex=False)]
    for cell in multi["antecedents"]:
        assert all(part for part in cell.split(ITEM_SEP))  # no empty fragments
