"""Contract tests for the committed rules artifact, plus full-dataset checks.

The artifact tests protect the DEPLOYED APP: streamlit_app.py assumes rules.csv
has certain columns, clean values, and the '|' separator. If a rebuild ever
violates that contract, these fail before a broken artifact reaches production.

The integration tests re-run the pipeline invariants on the real 540k-row data.
They skip automatically when the raw CSV is absent (it is git-ignored), so the
suite still passes on a fresh clone / CI runner.
"""

from pathlib import Path

import pandas as pd
import pytest

from mining import ITEM_SEP

ROOT = Path(__file__).resolve().parent.parent
RULES_CSV = ROOT / "data" / "rules.csv"
RAW_CSV = ROOT / "data" / "online_retail_raw.csv"

APP_REQUIRED_COLUMNS = {
    "antecedents", "consequents", "antecedent support", "consequent support",
    "support", "confidence", "lift", "leverage",
}


# ---------------------------------------------------------------- artifact contract

def test_rules_artifact_exists_and_has_required_columns():
    assert RULES_CSV.exists(), "data/rules.csv is missing — run src/build_rules.py"
    rules = pd.read_csv(RULES_CSV)
    assert APP_REQUIRED_COLUMNS <= set(rules.columns)
    assert len(rules) > 0


def test_rules_artifact_values_are_sane():
    rules = pd.read_csv(RULES_CSV)
    assert rules[list(APP_REQUIRED_COLUMNS)].notna().all().all()
    for col in ("support", "confidence", "antecedent support", "consequent support"):
        assert rules[col].between(0, 1).all()
    # mining floors that the app's sliders rely on
    assert (rules["support"] >= 0.01).all()
    assert (rules["confidence"] >= 0.10).all()
    # identity must survive the CSV round-trip
    assert ((rules["lift"] - rules["confidence"] / rules["consequent support"]).abs() < 1e-9).all()


def test_rules_artifact_itemsets_parse_cleanly():
    rules = pd.read_csv(RULES_CSV)
    for col in ("antecedents", "consequents"):
        parts = rules[col].str.split(ITEM_SEP, regex=False).explode()
        assert (parts.str.len() > 0).all(), f"empty item after splitting {col}"
        # the separator itself must never appear inside an item name
        assert not parts.str.contains(r"\|", regex=True).any()


# ---------------------------------------------------------------- full-data integration

needs_raw = pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not downloaded")


@pytest.fixture(scope="module")
def real_basket():
    from preprocess import build_basket_matrix, clean, load_raw
    cleaned, _ = clean(load_raw(str(RAW_CSV)))
    return cleaned, build_basket_matrix(cleaned)


@needs_raw
def test_real_pipeline_reproduces_known_totals(real_basket):
    cleaned, basket = real_basket
    assert len(cleaned) == 527_725
    assert basket.shape == (19_773, 3_986)


@needs_raw
def test_real_matrix_true_cells_equal_unique_pairs(real_basket):
    cleaned, basket = real_basket
    assert int(basket.values.sum()) == cleaned[["InvoiceNo", "Description"]].drop_duplicates().shape[0]
