"""Shared test fixtures.

The synthetic dataset is small enough to verify BY HAND. After cleaning it must
reduce to exactly these 5 baskets over 3 products:

    B 1001: BREAD, BUTTER
    B 1002: BREAD, BUTTER, MILK
    B 1003: BREAD, MILK          (BREAD arrives as ' bread ' -> tests normalization)
    B 1004: BUTTER
    B 1005: BREAD, BUTTER        (BREAD listed twice -> tests presence-not-quantity)

Hand-computed ground truth used across the tests (5 baskets):
    support(BREAD)=4/5   support(BUTTER)=4/5   support(MILK)=2/5
    support(BREAD,BUTTER)=3/5   support(BREAD,MILK)=2/5   support(BUTTER,MILK)=1/5
    rule BREAD->BUTTER: confidence=0.6/0.8=0.75, lift=0.75/0.8=0.9375  (lift < 1!)
    rule BREAD->MILK:   confidence=0.4/0.8=0.50, lift=0.50/0.4=1.25

The raw frame also carries one of EVERY junk type the cleaner must remove:
a return invoice (C-prefixed, negative qty), a POST line, a zero-price line,
a NaN description, and a whitespace-only description.
"""

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def raw_synthetic() -> pd.DataFrame:
    ts = pd.Timestamp("2011-06-01 10:00:00")
    rows = [
        # InvoiceNo, StockCode, Description, Quantity, UnitPrice, CustomerID
        # --- valid purchases (survive cleaning) --------------------------------
        ("1001", "10001", "BREAD",        1, 1.00, 111.0),
        ("1001", "10002", "BUTTER",       2, 2.00, 111.0),
        ("1002", "10001", "BREAD",        1, 1.00, 222.0),
        ("1002", "10002", "BUTTER",       1, 2.00, 222.0),
        ("1002", "10003", "MILK",         3, 0.50, 222.0),
        ("1003", "10001", " bread ",      1, 1.00, None),   # messy text + missing customer: must be KEPT
        ("1003", "10003", "MILK",         1, 0.50, None),
        ("1004", "10002", "BUTTER",       5, 2.00, 333.0),
        ("1005", "10001", "BREAD",        1, 1.00, 444.0),
        ("1005", "10001", "BREAD",        4, 1.00, 444.0),  # duplicate line: matrix cell must stay a single True
        ("1005", "10002", "BUTTER",       1, 2.00, 444.0),
        # --- junk (every row below must be removed) ----------------------------
        ("C1006", "10001", "BREAD",      -3, 1.00, 111.0),  # return / cancellation
        ("1001",  "POST",  "POSTAGE",     1, 5.00, 111.0),  # non-product stock code
        ("1002",  "10001", "BREAD",       1, 0.00, 222.0),  # zero price
        ("1004",  "10003", None,          1, 0.50, 333.0),  # blank description
        ("1004",  "10003", "   ",         1, 0.50, 333.0),  # whitespace-only description
    ]
    df = pd.DataFrame(
        rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID"]
    )
    df["InvoiceDate"] = ts
    df["Country"] = "United Kingdom"
    return df


# ground truth constants shared by the test modules
CLEAN_BASKETS = {
    "1001": {"BREAD", "BUTTER"},
    "1002": {"BREAD", "BUTTER", "MILK"},
    "1003": {"BREAD", "MILK"},
    "1004": {"BUTTER"},
    "1005": {"BREAD", "BUTTER"},
}
N_JUNK_ROWS = 5
