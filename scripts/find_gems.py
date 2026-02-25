#!/usr/bin/env python3
"""
Consolidated screening pipeline: scrape exchange listings + filter by fundamentals.

Two-phase internal pipeline:
  Phase 1 (scrape): Download ticker listings from configured exchanges
  Phase 2 (filter): Fetch yfinance financials and apply hard filters

Modes:
  Default:        Run both phases in-memory (scrape → filter → output)
  --scrape-only:  Only scrape exchanges, output raw CSV, skip filtering
  --filter-only:  Skip scraping, filter from existing CSV file

Replaces the manual chaining of ticker_scraper.py + filter_tickers.py.
Both original scripts remain untouched for backward compatibility.
"""

import argparse
import io
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# yfinance logs every 404 to stderr at ERROR level — suppress since missing
# tickers are expected and handled by returning None in _process_row.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# --- CONSTANTS ---
DEFAULT_CONFIG_PATH = "config/exchanges.json"
DEFAULT_WORKERS = 4
BATCH_SIZE = 50

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
]

# Currencies used by configured exchanges (fetched once per scan)
_FX_CURRENCIES = [
    "CAD",
    "GBP",
    "EUR",
    "CHF",
    "SEK",
    "NOK",
    "JPY",
    "HKD",
    "TWD",
    "SGD",
    "MYR",
    "IDR",
    "AUD",
    "NZD",
    "THB",
    "INR",
    "KRW",
]

SCRAPE_COLUMNS = [
    "Country",
    "Exchange",
    "YF_Ticker",
    "Ticker_Raw",
    "Company",
    "Sector",
    "Currency",
    "Lot_Size",
    "Listing_Code",
]

ENRICHED_COLUMNS = [
    "YF_Ticker",
    "Company_YF",
    "P/E",
    "Forward_PE",
    "Debt_to_Equity",
    "Net_Debt_to_Equity",
    "OCF_Yield",
    "OCF_NI_Ratio",
    "ROE",
    "ROA",
    "Operating_Cash_Flow",
    "Free_Cash_Flow",
    "Net_Income",
    "Total_Debt",
    "Total_Cash",
    "Revenue_Years_Positive",
    "Market_Cap",
    "Market_Cap_USD",
    "Avg_Volume",
    "Daily_Turnover_USD",
    "Analyst_Coverage",
    "YF_Sector",
    "YF_Industry",
    "Price",
    "Currency_YF",
    "Country",
    "Exchange",
    "Ticker_Raw",
]


# ============================================================
# FX helpers — fetch once per scan, reuse across all tickers
# ============================================================


def _fetch_one_fx_rate(currency):
    """Fetch a single FX rate (currency → USD). Returns (currency, rate|None)."""
    try:
        t = yf.Ticker(f"{currency}USD=X")
        price = getattr(t.fast_info, "last_price", None)
        if not price:
            price = t.info.get("regularMarketPrice")
        if price and price > 0:
            return currency, float(price)
    except Exception:
        pass
    return currency, None


def _fetch_fx_rates():
    """Fetch live FX rates from yfinance in parallel. Called once per scan."""
    rates = {"USD": 1.0}
    with ThreadPoolExecutor(max_workers=5) as executor:
        for cur, rate in executor.map(_fetch_one_fx_rate, _FX_CURRENCIES):
            if rate is not None:
                rates[cur] = rate
    # Fill gaps from fallback table
    try:
        from src.fx_normalization import FALLBACK_RATES_TO_USD

        for cur in _FX_CURRENCIES:
            if cur not in rates and cur in FALLBACK_RATES_TO_USD:
                rates[cur] = FALLBACK_RATES_TO_USD[cur]
    except ImportError:
        pass
    return rates


def _to_usd(value, currency, fx_rates):
    """Convert *value* in *currency* to USD. Returns None for unknown currencies."""
    if value is None or currency is None:
        return None
    # Normalize GBp (pence) → GBP
    if currency == "GBp":
        currency = "GBP"
        value = value / 100.0
    rate = fx_rates.get(currency)
    if rate is None:
        return None
    return value * rate


# ============================================================
# Phase 1: Scrape exchanges (from ticker_scraper.py)
# ============================================================


_DEFAULT_TIMEOUT = (10, 30)  # (connect, read) seconds


class _TimeoutAdapter(requests.adapters.HTTPAdapter):
    """Enforce a default timeout on every request made through the session."""

    def __init__(self, timeout=_DEFAULT_TIMEOUT, **kwargs):
        self._timeout = timeout
        super().__init__(**kwargs)

    def send(self, request, **kwargs):
        kwargs.setdefault("timeout", self._timeout)
        return super().send(request, **kwargs)


def _get_session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
    )
    adapter = _TimeoutAdapter()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _check_deps():
    import importlib.util

    missing = []
    for pkg in ("openpyxl", "xlrd", "lxml"):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        print(f"WARNING: Missing optional deps: {', '.join(missing)}", file=sys.stderr)
        print(f"Run: poetry add {' '.join(missing)}", file=sys.stderr)


def _find_col_fuzzy(df, target):
    if not target:
        return None
    target_clean = str(target).strip().lower()
    if target in df.columns:
        return target
    for col in df.columns:
        if str(col).strip().lower() == target_clean:
            return col
    return None


def _standardize_dataframe(df, config):
    params = config.get("params", {})
    rename_dict = {}

    ticker_col = params.get("ticker_col")
    actual_ticker = _find_col_fuzzy(df, ticker_col)
    if actual_ticker:
        rename_dict[actual_ticker] = "Ticker_Raw"

    name_col = params.get("name_col")
    actual_name = _find_col_fuzzy(df, name_col)
    if actual_name:
        rename_dict[actual_name] = "Company"

    col_map = params.get("col_map", {})
    for std_col, source_col in col_map.items():
        actual_source = _find_col_fuzzy(df, source_col)
        if actual_source:
            rename_dict[actual_source] = std_col

    df = df.rename(columns=rename_dict)
    df["Country"] = config["country"]
    df["Exchange"] = config["exchange_name"]

    for col in SCRAPE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df


def _generate_yf_ticker(row, config):
    if pd.isna(row.get("Ticker_Raw")) or str(row.get("Ticker_Raw")).strip() == "":
        return None

    raw = str(row["Ticker_Raw"]).strip()
    suffix = config["yahoo_suffix"]

    if config.get("params", {}).get("clean_rule") == "pad_4_digits":
        try:
            raw = raw.split(".")[0].zfill(4)
        except Exception:
            pass

    if suffix == "dynamic":
        suffix_map = config["params"].get("suffix_map", {})
        market_col = config["params"].get("market_col")
        if market_col and market_col in row and pd.notna(row[market_col]):
            market_val = str(row[market_col])
            for key, sfx in suffix_map.items():
                if key.lower() in market_val.lower():
                    return f"{raw}{sfx}"
        # No market match → drop rather than emit unsuffixed ticker (guaranteed 404)
        return None

    # Strip trailing dashes from raw mnemonics before appending suffix.
    # Some exchange files (e.g. LSE SETS) tag special-status securities with a
    # trailing dash (e.g. "JD-", "RM-"). Yahoo Finance uses the plain ticker.
    raw = raw.rstrip("-")
    if not raw:
        return None

    return f"{raw}{suffix}"


def _handle_download_json(config, session):
    response = session.get(config["source_url"])
    response.raise_for_status()
    data = response.json()
    params = config.get("params", {})
    root = params.get("root_key")
    if root:
        if root in data:
            data = data[root]
        else:
            raise ValueError(f"JSON root key '{root}' not found")
    if not isinstance(data, list):
        raise ValueError(f"JSON data is {type(data)}, expected list")
    return pd.DataFrame(data)


def _handle_download_csv(config, session):
    response = session.get(config["source_url"])
    response.raise_for_status()
    params = config["params"]
    skip = params.get("skip_rows", 0)
    sep = params.get("delimiter", ",")

    for enc in ["utf-8", "latin1", "cp1252", "utf-16"]:
        try:
            return pd.read_csv(
                io.BytesIO(response.content),
                sep=sep,
                skiprows=skip,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
            )
        except (UnicodeDecodeError, Exception):
            continue

    raise ValueError("CSV parsing failed (check delimiter or encoding)")


def _handle_download_excel(config, session):
    response = session.get(config["source_url"])
    response.raise_for_status()
    params = config["params"]
    sheet = params.get("sheet_name", 0)
    skip = params.get("skip_rows", 0)
    return pd.read_excel(io.BytesIO(response.content), sheet_name=sheet, skiprows=skip)


def _handle_scrape_html(config, session):
    response = session.get(config["source_url"])
    response.raise_for_status()

    if "<table" not in response.text.lower():
        raise ValueError("No HTML tables found (likely JS-rendered page)")

    params = config["params"]
    idx = params.get("table_index", 0)
    dfs = pd.read_html(io.StringIO(response.text), flavor=["lxml", "html5lib", "bs4"])

    if len(dfs) > idx:
        primary_df = dfs[idx]
        if _find_col_fuzzy(primary_df, params.get("ticker_col")):
            return primary_df

    target_col = params.get("ticker_col")
    if target_col:
        for df in dfs:
            if _find_col_fuzzy(df, target_col):
                return df

    raise ValueError(f"Table containing '{target_col}' not found")


_HANDLERS = {
    "download_csv": _handle_download_csv,
    "download_excel": _handle_download_excel,
    "scrape_html": _handle_scrape_html,
    "download_json": _handle_download_json,
}


def _apply_filters(df, config):
    """Apply positive filter and exclude_filter from exchange config to a DataFrame.

    Returns the filtered DataFrame (may be empty).
    """
    # Apply positive filter (keep rows matching column=value)
    filter_rules = config.get("params", {}).get("filter", {})
    if filter_rules:
        for col, value in filter_rules.items():
            actual_col = _find_col_fuzzy(df, col)
            if actual_col:
                df = df[df[actual_col].astype(str).str.strip() == str(value)]

    # Apply negative filter (exclude rows matching column pattern)
    exclude_rules = config.get("params", {}).get("exclude_filter", {})
    if exclude_rules:
        for col, values in exclude_rules.items():
            actual_col = _find_col_fuzzy(df, col)
            if actual_col:
                if isinstance(values, list):
                    df = df[~df[actual_col].astype(str).str.strip().isin(values)]
                else:
                    df = df[
                        ~df[actual_col]
                        .astype(str)
                        .str.contains(str(values), case=False, na=False)
                    ]

    return df


def scrape_exchanges(config: dict, *, exclude_us: bool = True) -> pd.DataFrame:
    """Scrape all configured exchanges. Returns DataFrame with YF_Ticker column.

    Skips US exchanges by default (config 'country' field = 'United States').
    """
    _check_deps()
    session = _get_session()
    all_dfs = []

    print(f"Loaded {config['meta']['description']}", file=sys.stderr)

    for ex in config["exchanges"]:
        country = ex.get("country", "")
        if exclude_us and country.lower() == "united states":
            print(f"Skipping {ex['exchange_name']} (US excluded)", file=sys.stderr)
            continue

        if not ex.get("enabled", True):
            print(f"Skipping {ex['exchange_name']} (disabled)", file=sys.stderr)
            continue

        print(
            f"Processing {country} ({ex['exchange_name']})...", file=sys.stderr, end=" "
        )

        handler = _HANDLERS.get(ex["method"])
        if not handler:
            print(f"Unknown method: {ex['method']}", file=sys.stderr)
            continue

        try:
            df = handler(ex, session)

            if df is None or df.empty:
                print("Empty DataFrame", file=sys.stderr)
                continue

            raw_cols = list(df.columns)

            df = _apply_filters(df, ex)

            if df.empty:
                print("Empty after filtering", file=sys.stderr)
                continue

            df = _standardize_dataframe(df, ex)
            df["YF_Ticker"] = df.apply(
                lambda r, _ex=ex: _generate_yf_ticker(r, _ex), axis=1
            )

            final_df = df[SCRAPE_COLUMNS].dropna(subset=["YF_Ticker"])
            all_dfs.append(final_df)

            if len(final_df) > 0:
                print(f"OK ({len(final_df)} rows)", file=sys.stderr)
            else:
                print(
                    f"OK (0 rows) - Check Config. Found columns: {raw_cols}",
                    file=sys.stderr,
                )

            time.sleep(1.0)

        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)

    if not all_dfs:
        print("No data extracted from any exchange.", file=sys.stderr)
        return pd.DataFrame(columns=SCRAPE_COLUMNS)

    master = pd.concat(all_dfs, ignore_index=True)
    master = master.drop_duplicates(subset=["YF_Ticker"])
    print(f"\nScraped {len(master)} unique tickers", file=sys.stderr)
    return master


# ============================================================
# Phase 2: Filter by financials (from filter_tickers.py)
# ============================================================


def _normalize_ticker(ticker):
    """Converts exchange formats to Yahoo format."""
    if not isinstance(ticker, str):
        return str(ticker)

    parts = ticker.split(".")
    if len(parts) > 2:
        symbol = "-".join(parts[:-1])
        suffix = parts[-1]
        return f"{symbol}.{suffix}"
    elif len(parts) == 2:
        p1, p2 = parts
        exchange_suffixes = {
            "V",
            "T",
            "L",
            "K",
            "S",
            "AX",
            "TO",
            "HK",
            "DE",
            "PA",
            "AS",
            "BR",
            "MI",
            "MC",
            "SW",
            "OL",
            "ST",
            "CO",
            "NZ",
            "JO",
            "KS",
            "KQ",
            "TW",
            "TWO",
            "SI",
            "LS",
            "KL",
            "BK",
            "JK",
            "NS",
            "BO",
        }
        if p2 in exchange_suffixes:
            return ticker
        if len(p2) == 1:
            return f"{p1}-{p2}"
    return ticker


def _process_row(row, *, fx_rates=None, min_mcap=None, min_volume=None, debug=False):
    """Fetch financials for a single ticker via yfinance.

    Early-exit filters (A/B/C) run BEFORE the expensive income_stmt fetch:
      A) Quote type must be EQUITY (reject ETFs, warrants, CBBCs)
      B) Market cap (USD) must exceed *min_mcap*
      C) Daily dollar volume (USD) must exceed *min_volume*
    """
    ticker_symbol = row.get("YF_Ticker")
    if pd.isna(ticker_symbol) or not ticker_symbol or str(ticker_symbol).strip() == "":
        return None

    yf_symbol = _normalize_ticker(str(ticker_symbol))
    if fx_rates is None:
        fx_rates = {"USD": 1.0}

    max_retries = 4
    for attempt in range(max_retries + 1):
        time.sleep(random.uniform(0.5, 1.5))

        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            if not info:
                if debug:
                    print(f"[DEBUG] {yf_symbol}: Empty info", file=sys.stderr)
                return None

            if (
                "regularMarketPrice" not in info
                and "currentPrice" not in info
                and "trailingPE" not in info
            ):
                if debug:
                    print(
                        f"[DEBUG] {yf_symbol}: No price/PE data found", file=sys.stderr
                    )
                return None

            # --- Filter A: Quote type guard (ETFs, warrants, CBBCs) ---
            quote_type = info.get("quoteType")
            if quote_type and quote_type != "EQUITY":
                if debug:
                    print(
                        f"[SKIP] {yf_symbol}: quoteType={quote_type} (not EQUITY)",
                        file=sys.stderr,
                    )
                return None

            # Extract price/currency early (needed for Filters B & C)
            currency = info.get("currency")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            market_cap = info.get("marketCap")
            avg_volume = info.get("averageVolume")

            # --- Filter B: Minimum market cap (USD) ---
            mcap_usd = None
            if market_cap is not None and currency:
                mcap_usd = _to_usd(market_cap, currency, fx_rates)
            if (
                min_mcap
                and min_mcap > 0
                and mcap_usd is not None
                and mcap_usd < min_mcap
            ):
                if debug:
                    print(
                        f"[SKIP] {yf_symbol}: Micro-cap ${mcap_usd:,.0f} < ${min_mcap:,.0f}",
                        file=sys.stderr,
                    )
                return None

            # --- Filter C: Minimum daily dollar volume (USD) ---
            turnover_usd = None
            if avg_volume and price and currency:
                local_turnover = avg_volume * price
                # Pence correction for .L tickers (same as liquidity_calculation_tool)
                if yf_symbol.endswith(".L"):
                    local_turnover = local_turnover / 100.0
                turnover_usd = _to_usd(local_turnover, currency, fx_rates)
            if (
                min_volume
                and min_volume > 0
                and turnover_usd is not None
                and turnover_usd < min_volume
            ):
                if debug:
                    print(
                        f"[SKIP] {yf_symbol}: Low volume ${turnover_usd:,.0f} < ${min_volume:,.0f}",
                        file=sys.stderr,
                    )
                return None

            # --- Populate standard fields ---
            row["Company_YF"] = info.get("longName") or info.get("shortName")
            row["P/E"] = info.get("trailingPE")
            row["Forward_PE"] = info.get("forwardPE")
            row["ROE"] = info.get("returnOnEquity")
            row["ROA"] = info.get("returnOnAssets")
            row["Debt_to_Equity"] = info.get("debtToEquity")
            row["Operating_Cash_Flow"] = info.get("operatingCashflow")
            row["Free_Cash_Flow"] = info.get("freeCashflow")
            row["Net_Income"] = info.get("netIncomeToCommon")
            row["Total_Debt"] = info.get("totalDebt")
            row["Total_Cash"] = info.get("totalCash")
            row["Market_Cap"] = market_cap
            row["Market_Cap_USD"] = mcap_usd
            row["Avg_Volume"] = avg_volume
            row["Daily_Turnover_USD"] = turnover_usd
            row["Analyst_Coverage"] = info.get("numberOfAnalystOpinions")

            if (
                row["Operating_Cash_Flow"]
                and row["Market_Cap"]
                and row["Market_Cap"] > 0
            ):
                row["OCF_Yield"] = row["Operating_Cash_Flow"] / row["Market_Cap"]
            else:
                row["OCF_Yield"] = None

            # Earnings quality: OCF / Net Income
            ocf = row["Operating_Cash_Flow"]
            ni = row["Net_Income"]
            if ocf and ni and ni > 0:
                row["OCF_NI_Ratio"] = ocf / ni
            else:
                row["OCF_NI_Ratio"] = None

            # Net Debt/Equity: (Debt - Cash) / Equity
            de = row["Debt_to_Equity"]
            total_debt = row["Total_Debt"]
            total_cash = row["Total_Cash"]
            if de and de > 0 and total_debt and total_cash is not None:
                equity = total_debt / (de / 100.0)
                if equity > 0:
                    row["Net_Debt_to_Equity"] = (
                        (total_debt - total_cash) / equity * 100.0
                    )
                else:
                    row["Net_Debt_to_Equity"] = None
            else:
                row["Net_Debt_to_Equity"] = None

            # Revenue history: count annual periods with positive revenue
            row["Revenue_Years_Positive"] = None
            try:
                income_stmt = ticker.income_stmt
                if income_stmt is not None and not income_stmt.empty:
                    for label in ["Total Revenue", "Operating Revenue"]:
                        if label in income_stmt.index:
                            revenues = income_stmt.loc[label].dropna()
                            row["Revenue_Years_Positive"] = int((revenues > 0).sum())
                            break
            except Exception:
                pass

            row["YF_Sector"] = info.get("sector")
            row["YF_Industry"] = info.get("industry")
            row["Price"] = price
            row["Currency_YF"] = currency

            return row

        except Exception as e:
            str_e = str(e)

            if "404" in str_e and "Not Found" in str_e:
                if debug:
                    print(f"[DEBUG] {yf_symbol}: 404 Not Found", file=sys.stderr)
                return None

            if "429" in str_e or "Too Many Requests" in str_e or "500" in str_e:
                if attempt < max_retries:
                    is_rate_limit = "429" in str_e or "Too Many Requests" in str_e
                    base_wait = 20 if is_rate_limit else 5
                    wait_time = (base_wait * (attempt + 1)) + random.uniform(1, 5)
                    if debug:
                        print(
                            f"[RETRY] {yf_symbol}: Sleeping {wait_time:.1f}s",
                            file=sys.stderr,
                        )
                    time.sleep(wait_time)
                    continue

            if debug:
                print(f"[DEBUG] {yf_symbol}: Error {str_e}", file=sys.stderr)
            return None

    return None


def _safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _passes_filters(
    row,
    *,
    max_pe,
    min_roe,
    min_roa,
    max_de,
    min_ocf_ni_ratio=0.8,
    min_revenue_years=3,
    max_coverage=30,
    ocf_waiver=True,
    debug=False,
):
    """Apply hard financial filters to an enriched row."""
    if not row:
        return False

    ticker = row.get("YF_Ticker", "?")

    # --- P/E ---
    pe = _safe_float(row.get("P/E"))
    if pe is None:
        if debug:
            print(f"[SKIP] {ticker}: Missing P/E", file=sys.stderr)
        return False
    if pe > max_pe:
        if debug:
            print(f"[SKIP] {ticker}: P/E {pe} > {max_pe}", file=sys.stderr)
        return False

    # --- Profitability (ROE OR ROA) ---
    roe = _safe_float(row.get("ROE"))
    roa = _safe_float(row.get("ROA"))
    roe_threshold = min_roe / 100.0
    roa_threshold = min_roa / 100.0

    has_good_roe = roe is not None and roe > roe_threshold
    has_good_roa = roa is not None and roa > roa_threshold

    if not (has_good_roe or has_good_roa):
        if debug:
            print(
                f"[SKIP] {ticker}: Low Profit (ROE={roe}, ROA={roa})", file=sys.stderr
            )
        return False

    # --- Leverage: Gross D/E with net debt fallback ---
    de = _safe_float(row.get("Debt_to_Equity"))
    if de is None:
        if debug:
            print(f"[SKIP] {ticker}: Missing D/E", file=sys.stderr)
        return False
    if de > max_de:
        # Fallback: net debt/equity = (Debt - Cash) / Equity
        net_de = _safe_float(row.get("Net_Debt_to_Equity"))
        if net_de is not None and net_de <= max_de:
            if debug:
                print(
                    f"[NOTE] {ticker}: Gross D/E {de}% > {max_de}%, "
                    f"but Net D/E {net_de:.0f}% OK",
                    file=sys.stderr,
                )
        else:
            if debug:
                net_str = f"{net_de:.0f}%" if net_de is not None else "N/A"
                print(
                    f"[SKIP] {ticker}: High Debt (D/E={de}%, Net D/E={net_str})",
                    file=sys.stderr,
                )
            return False

    # --- Filter D: Maximum analyst coverage ---
    if max_coverage and max_coverage > 0:
        coverage = row.get("Analyst_Coverage")
        if coverage is not None:
            try:
                coverage = int(coverage)
                if coverage > max_coverage:
                    if debug:
                        print(
                            f"[SKIP] {ticker}: Too many analysts ({coverage} > {max_coverage})",
                            file=sys.stderr,
                        )
                    return False
            except (ValueError, TypeError):
                pass  # fail-open: non-numeric coverage → skip check

    # --- Operating Cash Flow > 0 (with Filter E: forward-PE recovery waiver) ---
    ocf = _safe_float(row.get("Operating_Cash_Flow"))
    if ocf is None or ocf <= 0:
        # Filter E: waiver for transient OCF dips when forward PE signals recovery
        if ocf_waiver and ocf is not None and ocf <= 0:
            fwd_pe = _safe_float(row.get("Forward_PE"))
            trailing_pe = _safe_float(row.get("P/E"))
            if (
                fwd_pe is not None
                and trailing_pe is not None
                and fwd_pe < trailing_pe
                and fwd_pe < 15
            ):
                if debug:
                    print(
                        f"[WAIVER] {ticker}: OCF negative but forward PE {fwd_pe:.1f} "
                        f"< trailing {trailing_pe:.1f} and < 15 → recovery expected",
                        file=sys.stderr,
                    )
                # Fall through to remaining checks (don't return False)
            else:
                if debug:
                    print(f"[SKIP] {ticker}: Negative/No OCF", file=sys.stderr)
                return False
        else:
            if debug:
                print(f"[SKIP] {ticker}: Negative/No OCF", file=sys.stderr)
            return False

    # --- Earnings quality: OCF / Net Income (when NI positive) ---
    ni = _safe_float(row.get("Net_Income"))
    if ni is not None and ni > 0:
        ocf_ni = _safe_float(row.get("OCF_NI_Ratio"))
        if ocf_ni is not None and ocf_ni < min_ocf_ni_ratio:
            if debug:
                print(
                    f"[SKIP] {ticker}: Low earnings quality "
                    f"(OCF/NI={ocf_ni:.2f} < {min_ocf_ni_ratio})",
                    file=sys.stderr,
                )
            return False

    # --- Revenue history: 3+ years of positive revenue ---
    rev_years = row.get("Revenue_Years_Positive")
    if rev_years is not None:
        rev_years = int(rev_years)
        if rev_years < min_revenue_years:
            if debug:
                print(
                    f"[SKIP] {ticker}: Insufficient revenue history "
                    f"({rev_years}yr positive < {min_revenue_years}yr required)",
                    file=sys.stderr,
                )
            return False

    if debug:
        print(f"[KEEP] {ticker}: P/E={pe}, D/E={de}%, ROE={roe}", file=sys.stderr)
    return True


def _load_csv_robust(filepath):
    """Robust CSV loader that handles bad lines and encoding issues."""
    try:
        df = pd.read_csv(filepath, on_bad_lines="skip", engine="python")

        if "YF_Ticker" in df.columns and len(df) > 0:
            sample_ticker_col = str(df["YF_Ticker"].iloc[0])
            sample_index = str(df.index[0])

            if " " in sample_ticker_col and (
                "." in sample_index or sample_index.isupper()
            ):
                print(
                    "Warning: Detected CSV misalignment (Ticker in index). Fixing...",
                    file=sys.stderr,
                )
                df.reset_index(inplace=True)
                index_col = df.columns[0]
                df["YF_Ticker"] = df[index_col]

        if "YF_Ticker" not in df.columns:
            if "Ticker" in df.columns:
                df.rename(columns={"Ticker": "YF_Ticker"}, inplace=True)
            else:
                df.rename(columns={df.columns[0]: "YF_Ticker"}, inplace=True)
                print(
                    "Warning: 'YF_Ticker' column not found, using first column as ticker.",
                    file=sys.stderr,
                )

        return df
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_and_filter(
    tickers_df: pd.DataFrame,
    *,
    max_pe: float = 18.0,
    min_roe: float = 13.0,
    min_roa: float = 6.0,
    max_de: float = 150.0,
    min_mcap: float = 50_000_000,
    min_volume: float = 100_000,
    max_coverage: int = 30,
    ocf_waiver: bool = True,
    workers: int = 4,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch yfinance financials and apply hard filters.

    Returns (passing_df, all_enriched_df).
    """
    records = tickers_df.to_dict("records")
    total = len(records)

    # Fetch FX rates once (parallel, ~2s)
    print("Fetching live FX rates...", file=sys.stderr, end=" ")
    fx_rates = _fetch_fx_rates()
    print(f"OK ({len(fx_rates)} currencies)", file=sys.stderr)

    print(
        f"Scanning {total} tickers with {workers} workers...",
        file=sys.stderr,
    )
    criteria_parts = [
        f"P/E<{max_pe}",
        f"ROE>{min_roe}%/ROA>{min_roa}%",
        f"D/E<{max_de}%",
        "OCF>0",
    ]
    if min_mcap and min_mcap > 0:
        criteria_parts.append(f"MCap>${min_mcap/1e6:.0f}M")
    if min_volume and min_volume > 0:
        criteria_parts.append(f"Vol>${min_volume/1e3:.0f}K")
    if max_coverage and max_coverage > 0:
        criteria_parts.append(f"Analysts<={max_coverage}")
    print(f"Criteria: {', '.join(criteria_parts)}", file=sys.stderr)

    passing = []
    all_enriched = []
    processed_count = 0
    start_time = time.time()

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_row = {
                executor.submit(
                    _process_row,
                    row,
                    fx_rates=fx_rates,
                    min_mcap=min_mcap,
                    min_volume=min_volume,
                    debug=debug,
                ): row
                for row in records
            }

            for future in as_completed(future_to_row):
                processed_count += 1
                try:
                    data = future.result()
                    if data is not None:
                        all_enriched.append(data)

                        if _passes_filters(
                            data,
                            max_pe=max_pe,
                            min_roe=min_roe,
                            min_roa=min_roa,
                            max_de=max_de,
                            max_coverage=max_coverage,
                            ocf_waiver=ocf_waiver,
                            debug=debug,
                        ):
                            passing.append(data)

                    if processed_count % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        n_pass = len(passing)
                        print(
                            f"Progress: {processed_count}/{total} ({rate:.1f} t/s, {n_pass} passing)",
                            file=sys.stderr,
                        )
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("\nInterrupted! Returning partial results...", file=sys.stderr)

    passing_df = (
        pd.DataFrame(passing) if passing else pd.DataFrame(columns=ENRICHED_COLUMNS)
    )
    enriched_df = (
        pd.DataFrame(all_enriched)
        if all_enriched
        else pd.DataFrame(columns=ENRICHED_COLUMNS)
    )

    print(
        f"\nFilter complete: {len(passing_df)}/{total} tickers passed",
        file=sys.stderr,
    )

    return passing_df, enriched_df


# ============================================================
# Output
# ============================================================


def write_outputs(
    filtered_df: pd.DataFrame, output_path: str, details_path: str | None = None
):
    """Write ticker-only file (one per line) + optional enriched CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tickers = filtered_df["YF_Ticker"].dropna().unique()

    with open(out, "w") as f:
        for t in sorted(tickers):
            f.write(f"{t}\n")

    print(f"Wrote {len(tickers)} tickers to {out}", file=sys.stderr)

    if details_path:
        det = Path(details_path)
        det.parent.mkdir(parents=True, exist_ok=True)
        available_cols = [c for c in ENRICHED_COLUMNS if c in filtered_df.columns]
        extra_cols = [c for c in filtered_df.columns if c not in available_cols]
        filtered_df[available_cols + extra_cols].to_csv(det, index=False)
        print(f"Wrote enriched details to {det}", file=sys.stderr)


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape exchange listings and filter by fundamentals (consolidated pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full pipeline (scrape + filter, ex-US)
  python scripts/find_gems.py --output scratch/gems.txt

  # With enriched CSV for debugging
  python scripts/find_gems.py --output scratch/gems.txt --details scratch/gems_details.csv

  # Scrape only (produce raw CSV, no yfinance filtering)
  python scripts/find_gems.py --scrape-only --output scratch/raw_tickers.csv

  # Filter from existing CSV (skip scraping)
  python scripts/find_gems.py --filter-only scratch/raw_tickers.csv --output scratch/gems.txt

  # Include US exchanges
  python scripts/find_gems.py --include-us --output scratch/gems.txt --debug
""",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output file. Ticker-only (.txt) in filter mode, CSV in scrape-only mode.",
    )
    parser.add_argument(
        "--details",
        help="Also write enriched CSV with financial metrics (filter mode only).",
    )
    parser.add_argument(
        "--configfile", default=DEFAULT_CONFIG_PATH, help="Exchange config JSON"
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape exchanges, output raw CSV, skip filtering.",
    )
    mode_group.add_argument(
        "--filter-only",
        metavar="FILE",
        help="Skip scraping, filter from existing CSV file.",
    )

    parser.add_argument(
        "--include-us",
        action="store_true",
        help="Include US exchanges (excluded by default)",
    )
    parser.add_argument(
        "--max-pe", type=float, default=18.0, help="Max P/E ratio (default: 18.0)"
    )
    parser.add_argument(
        "--min-roe", type=float, default=13.0, help="Min ROE %% (default: 13.0)"
    )
    parser.add_argument(
        "--min-roa", type=float, default=6.0, help="Min ROA %% (default: 6.0)"
    )
    parser.add_argument(
        "--max-de",
        type=float,
        default=150.0,
        help="Max D/E %% (default: 150.0, i.e. 1.5x)",
    )
    parser.add_argument(
        "--min-mcap",
        type=float,
        default=50_000_000,
        help="Min market cap in USD (default: 50000000, set 0 to disable)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=100_000,
        help="Min daily dollar volume in USD (default: 100000, set 0 to disable)",
    )
    parser.add_argument(
        "--max-coverage",
        type=int,
        default=30,
        help="Max analyst coverage count (default: 30, set 0 to disable)",
    )
    parser.add_argument(
        "--no-ocf-waiver",
        action="store_true",
        help="Disable forward-PE recovery waiver for negative OCF",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="ThreadPoolExecutor concurrency (default: 4)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show skip reasons per ticker"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Mode: scrape-only ---
    if args.scrape_only:
        with open(args.configfile) as f:
            config = json.load(f)

        scraped_df = scrape_exchanges(config, exclude_us=not args.include_us)

        if scraped_df.empty:
            print("No tickers scraped.", file=sys.stderr)
            sys.exit(1)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        scraped_df.to_csv(out, index=False)
        print(f"Saved {len(scraped_df)} rows to {out}", file=sys.stderr)
        return

    # --- Mode: filter-only ---
    if args.filter_only:
        tickers_df = _load_csv_robust(args.filter_only)

        passing_df, enriched_df = fetch_and_filter(
            tickers_df,
            max_pe=args.max_pe,
            min_roe=args.min_roe,
            min_roa=args.min_roa,
            max_de=args.max_de,
            min_mcap=args.min_mcap,
            min_volume=args.min_volume,
            max_coverage=args.max_coverage,
            ocf_waiver=not args.no_ocf_waiver,
            workers=args.workers,
            debug=args.debug,
        )

        if passing_df.empty:
            print("No tickers passed filters.", file=sys.stderr)
            sys.exit(1)

        write_outputs(passing_df, args.output, args.details)
        return

    # --- Default mode: scrape + filter ---
    with open(args.configfile) as f:
        config = json.load(f)

    scraped_df = scrape_exchanges(config, exclude_us=not args.include_us)

    if scraped_df.empty:
        print("No tickers scraped. Aborting.", file=sys.stderr)
        sys.exit(1)

    passing_df, enriched_df = fetch_and_filter(
        scraped_df,
        max_pe=args.max_pe,
        min_roe=args.min_roe,
        min_roa=args.min_roa,
        max_de=args.max_de,
        min_mcap=args.min_mcap,
        min_volume=args.min_volume,
        max_coverage=args.max_coverage,
        ocf_waiver=not args.no_ocf_waiver,
        workers=args.workers,
        debug=args.debug,
    )

    if passing_df.empty:
        print("No tickers passed filters.", file=sys.stderr)
        sys.exit(1)

    write_outputs(passing_df, args.output, args.details)


if __name__ == "__main__":
    main()
