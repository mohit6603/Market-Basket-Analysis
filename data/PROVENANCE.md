# Dataset provenance — Online Retail

| | |
|---|---|
| **Name** | Online Retail |
| **Source** | UCI Machine Learning Repository (ID 352) |
| **Download URL** | https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx |
| **Downloaded** | 2026-07-01 |
| **Raw file** | `data/Online Retail.xlsx` (~23 MB, single sheet) |
| **Cached CSV** | `data/online_retail_raw.csv` (generated once from the xlsx for faster reloads) |
| **Shape** | 541,909 rows × 8 columns |
| **Period** | 2010-12-01 → 2011-12-09 (~1 year) |
| **License / citation** | Daqing Chen, Sai Liang Sain, Kun Guo (2012). *Data mining for the online retail industry.* Journal of Database Marketing & Customer Strategy Management, 19(3). |

## What it is
Real transactions from a UK-based, registered, non-store online retailer that sells
mainly all-occasion **giftware**. Many customers are wholesalers — which is why some
quantities are huge (hundreds per line).

## Grain
**One row = one product line on one invoice.** A single basket (invoice) spans
multiple rows. This is the **LONG** shape we will pivot to WIDE at Stage 5.

## Columns
| Column | Raw dtype | Meaning | Notes / quirks |
|---|---|---|---|
| `InvoiceNo` | object (str) | Invoice / transaction id | String, not int. A leading **`C`** marks a **cancellation/return**. |
| `StockCode` | object (str) | Product code | Alphanumeric (e.g. `85123A`). Some are **non-products**: `POST`, `DOT`, `M`, `BANK CHARGES`, `C2`, `PADS`, `S`, etc. |
| `Description` | object (str) | Product name | 1,454 blanks (0.27%). Some are manual-adjustment notes, not products. |
| `Quantity` | int64 | Units on this line | **Negative = return** (min −80,995). |
| `InvoiceDate` | datetime64 | Timestamp of the invoice | — |
| `UnitPrice` | float64 | Price per unit, GBP (£) | Some **0 or negative** (adjustments, min −11,062.06). |
| `CustomerID` | float64 | Customer id | **24.93% missing.** Float (not int) only because numpy ints can't hold NaN. |
| `Country` | object (str) | Customer country | 38 countries; ~91% United Kingdom. |

## Known issues to handle in cleaning (Stage 4)
- Returns / cancellations (negative `Quantity`, `C`-prefixed `InvoiceNo`).
- Missing `CustomerID` (~25%).
- Non-product line items (postage, fees, manual adjustments via special `StockCode`s).
- Blank / junk `Description`.
- Zero / negative `UnitPrice`.
