"""Clean the raw Online Retail data for market basket analysis.

Grain of the raw data: one row = one product line on one invoice (LONG format).
Cleaning keeps only rows that represent a *valid product purchase*, because those
are the only rows that can legitimately appear in a shopping basket.

Design decisions (be ready to defend these in an interview):

* Item identity = cleaned ``Description`` (human-readable), NOT ``StockCode``.
  Rules must be actionable to a merchandiser, so "{PARTY BUNTING} -> {...}" beats
  "{47566} -> {...}". Cost: two StockCodes with the same description merge into one
  item (rare after the other filters). Alternative: group by StockCode and attach a
  representative description for display -- more robust, more code; not worth it here.

* Keep rows with missing CustomerID (~25%). Baskets are grouped by ``InvoiceNo``, so
  CustomerID is never read. Dropping a quarter of the data for an unused column would
  be wasteful. (User decision.)

* Keep all 38 countries. The UK is already 90.7% of invoices, so the rules are
  UK-driven regardless; no arbitrary geographic cut. (User decision.)

* Remove only rows that are invalid AS A PURCHASE:
    - returns / cancellations   -> Quantity <= 0   (C-invoices are a subset of these)
    - non-purchase price rows    -> UnitPrice <= 0  (free gifts, manual adjustments)
    - non-product line items     -> StockCode not matching a product pattern
                                    (POST, DOT, M, C2, BANK CHARGES, AMAZONFEE, ...)
    - unnameable items           -> blank Description
"""

from __future__ import annotations

import re

import pandas as pd

# Real products are a 5-digit code with an optional letter suffix, e.g. 85123A.
# Everything else (POST, DOT, M, D, S, C2, BANK CHARGES, gift_0001_20, DCGSSGIRL...)
# is a fee / adjustment / non-product and must go.
PRODUCT_CODE_RE = re.compile(r"^\d{5}[A-Za-z]*$")

RAW_CSV = "data/online_retail_raw.csv"
CLEAN_CSV = "data/online_retail_clean.csv"


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    """Load the cached raw CSV with the dtypes that matter.

    InvoiceNo / StockCode are forced to str so the 'C' cancellation prefix and
    alphanumeric codes survive; InvoiceDate is parsed to datetime.
    """
    return pd.read_csv(
        path,
        dtype={"InvoiceNo": str, "StockCode": str},
        parse_dates=["InvoiceDate"],
    )


def _normalize_description(s: pd.Series) -> pd.Series:
    """Strip, collapse internal whitespace, and uppercase.

    Merges accidental variants like 'LUNCH BAG  BLACK SKULL.' (double space) and
    trailing-space duplicates into a single item label.
    """
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int, int]]]:
    """Apply the cleaning steps in order, returning the cleaned frame and a report.

    The report is a list of (step_label, rows_removed, rows_remaining) so the
    notebook / README can show exactly where every dropped row went.
    """
    df = df.copy()
    report: list[tuple[str, int, int]] = [("start", 0, len(df))]

    def drop(mask: pd.Series, label: str) -> None:
        nonlocal df
        before = len(df)
        df = df[~mask].copy()
        report.append((label, before - len(df), len(df)))

    # 1. Unnameable items -------------------------------------------------------
    drop(df["Description"].isna(), "drop blank Description")
    df["Description"] = _normalize_description(df["Description"])
    drop(df["Description"].eq(""), "drop empty-after-normalize")

    # 2. Returns / cancellations (negative or zero quantity) --------------------
    drop(df["Quantity"] <= 0, "drop Quantity <= 0 (returns)")

    # 3. Non-purchase prices ----------------------------------------------------
    drop(df["UnitPrice"] <= 0, "drop UnitPrice <= 0")

    # 4. Non-product line items (postage, fees, adjustments) --------------------
    is_product = df["StockCode"].str.match(PRODUCT_CODE_RE)
    drop(~is_product, "drop non-product StockCode")

    return df, report


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Reshape cleaned LONG data into a WIDE one-hot basket matrix.

    rows    = invoices (baskets)
    columns = products (by Description)
    cells   = True if the product appears in that basket, else False

    Why this construction (and not the common ``groupby.sum().unstack()``):

    * ``assign(_present=True)`` tags every existing (invoice, item) line as present.
    * ``.groupby([...])["_present"].any()`` collapses duplicate lines of the same
      item in one invoice to a single True -- this is the PRESENCE step. It throws
      away Quantity on purpose: association support cares only about "in the basket
      or not", never how many. (Bought 6? Still just True.)
    * ``.unstack(fill_value=False)`` spreads items across columns and fills every
      absent (invoice, item) combo with False. Because the source is already bool
      and the fill is False, the result stays ``bool`` end-to-end -- so we never
      build the ~630 MB int64 matrix that ``fill_value=0`` would create; the final
      matrix is ~79 MB.

    The output is exactly the boolean one-hot frame mlxtend's apriori/fpgrowth expect.
    """
    basket = (
        df.assign(_present=True)
        .groupby(["InvoiceNo", "Description"])["_present"]
        .any()
        .unstack(fill_value=False)
    )
    basket.columns.name = None  # drop the leftover 'Description' axis label
    return basket


def format_report(report: list[tuple[str, int, int]], df_start: pd.DataFrame, df_end: pd.DataFrame) -> str:
    """Human-readable before/after summary of a cleaning run."""
    lines = ["step                              removed    remaining"]
    for label, removed, remaining in report:
        lines.append(f"{label:<32} {removed:>8,}   {remaining:>10,}")
    start_n, end_n = report[0][2], report[-1][2]
    lines.append("-" * 56)
    lines.append(f"rows      : {start_n:,} -> {end_n:,}  (kept {end_n / start_n * 100:.1f}%)")
    lines.append(f"invoices  : {df_start['InvoiceNo'].nunique():,} -> {df_end['InvoiceNo'].nunique():,}")
    lines.append(f"products  : {df_start['Description'].nunique():,} -> {df_end['Description'].nunique():,} (by Description)")
    return "\n".join(lines)


if __name__ == "__main__":
    raw = load_raw()
    cleaned, report = clean(raw)
    print(format_report(report, raw, cleaned))
    cleaned.to_csv(CLEAN_CSV, index=False)
    print(f"\nwrote {CLEAN_CSV}  ({len(cleaned):,} rows)")
