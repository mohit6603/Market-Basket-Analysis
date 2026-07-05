"""Tests for cleaning and the LONG->WIDE basket pivot.

Strategy: run the real functions on the hand-checkable synthetic frame from
conftest.py and assert the exact outcome we computed on paper. The pivot is the
riskiest step of the whole project, so it gets the most assertions.
"""

import pandas as pd

from conftest import CLEAN_BASKETS, N_JUNK_ROWS
from preprocess import build_basket_matrix, clean


def test_clean_removes_every_junk_row_and_nothing_else(raw_synthetic):
    cleaned, report = clean(raw_synthetic)

    # exactly the 5 junk rows go; all 11 valid lines survive
    assert len(cleaned) == len(raw_synthetic) - N_JUNK_ROWS
    # the return invoice disappears entirely; the 5 real baskets remain
    assert set(cleaned["InvoiceNo"]) == set(CLEAN_BASKETS)
    # the report's row accounting must reconcile with the frame it returned
    assert report[-1][2] == len(cleaned)
    assert sum(removed for _, removed, _ in report) == N_JUNK_ROWS


def test_clean_keeps_rows_with_missing_customer_id(raw_synthetic):
    # design decision under test: baskets group by InvoiceNo, so missing
    # CustomerID must NOT cause a drop (invoice 1003 has only NaN customers)
    cleaned, _ = clean(raw_synthetic)
    assert "1003" in set(cleaned["InvoiceNo"])


def test_clean_normalizes_description_variants(raw_synthetic):
    # ' bread ' (messy) must merge into 'BREAD', not become a separate product
    cleaned, _ = clean(raw_synthetic)
    products = set(cleaned["Description"])
    assert products == {"BREAD", "BUTTER", "MILK"}


def test_basket_matrix_matches_hand_computed_baskets(raw_synthetic):
    cleaned, _ = clean(raw_synthetic)
    basket = build_basket_matrix(cleaned)

    # shape and dtype: 5 baskets x 3 products, strictly boolean, no NaN
    assert basket.shape == (len(CLEAN_BASKETS), 3)
    assert (basket.dtypes == bool).all()
    assert not basket.isna().any().any()

    # every cell equals the hand-written ground truth
    for invoice, items in CLEAN_BASKETS.items():
        row = basket.loc[invoice]
        assert set(row[row].index) == items, f"basket {invoice} wrong"


def test_basket_matrix_stores_presence_not_quantity(raw_synthetic):
    # invoice 1005 lists BREAD twice (qty 1 and 4): the cell must be a single
    # True, and the row sum must count ITEMS (2), not lines or units
    cleaned, _ = clean(raw_synthetic)
    basket = build_basket_matrix(cleaned)
    assert basket.loc["1005", "BREAD"] == True  # noqa: E712 (bool dtype, not truthiness)
    assert basket.loc["1005"].sum() == 2


def test_basket_matrix_true_cells_equal_unique_invoice_item_pairs(raw_synthetic):
    # the same invariant we used to verify the real 19,773x3,986 matrix
    cleaned, _ = clean(raw_synthetic)
    basket = build_basket_matrix(cleaned)
    unique_pairs = cleaned[["InvoiceNo", "Description"]].drop_duplicates().shape[0]
    assert int(basket.values.sum()) == unique_pairs
