#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf

# --- CONFIGURATION DEFAULTS ---
DEFAULT_WORKERS = 4
BATCH_SIZE = 50

# Output Column Order (Prioritize metrics for easy reading)
OUTPUT_ORDER = [
    "YF_Ticker",
    "Company_YF",
    "P/E",
    "Debt_to_Equity",
    "OCF_Yield",
    "ROE",
    "ROA",
    "Operating_Cash_Flow",
    "Free_Cash_Flow",
    "Market_Cap",
    "YF_Sector",
    "YF_Industry",
    "Price",
    "Currency_YF",
    "Country",
    "Exchange",
    "Ticker_Raw",  # Original meta at end
]


def normalize_ticker(ticker):
    """
    Converts exchange formats to Yahoo format.
    Handles distinct logic for US share classes vs International suffixes.
    """
    if not isinstance(ticker, str):
        return str(ticker)

    parts = ticker.split(".")
    if len(parts) > 2:
        symbol = "-".join(parts[:-1])
        suffix = parts[-1]
        return f"{symbol}.{suffix}"
    elif len(parts) == 2:
        p1, p2 = parts
        # Whitelist of suffixes that are definitely exchanges, not share classes
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
        }
        if p2 in exchange_suffixes:
            return ticker
        if len(p2) == 1:
            return f"{p1}-{p2}"
    return ticker


def process_row(row, debug=False):
    """
    Fetches financials for a row and merges them with existing data.
    """
    ticker_symbol = row.get("YF_Ticker")
    if pd.isna(ticker_symbol) or not ticker_symbol or str(ticker_symbol).strip() == "":
        return None

    yf_symbol = normalize_ticker(str(ticker_symbol))

    # Retry Logic for 429s/5xxs
    max_retries = 4
    for attempt in range(max_retries + 1):
        # Jitter to desynchronize threads
        time.sleep(random.uniform(0.5, 1.5))

        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            if not info:
                if debug:
                    print(f"[DEBUG] {yf_symbol}: Empty info", file=sys.stderr)
                return None

            # Check basic data presence (Price or PE must exist to be worth checking)
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

            # --- EXTRACT METRICS ---
            # Valuation & Profitability
            row["Company_YF"] = info.get("longName") or info.get("shortName")
            row["P/E"] = info.get("trailingPE")
            row["ROE"] = info.get("returnOnEquity")
            row["ROA"] = info.get("returnOnAssets")

            # Health (Debt & Cash Flow)
            # Yahoo returns Debt/Equity as a percentage (e.g. 150.5 = 1.5x)
            row["Debt_to_Equity"] = info.get("debtToEquity")
            row["Operating_Cash_Flow"] = info.get("operatingCashflow")
            row["Free_Cash_Flow"] = info.get("freeCashflow")
            row["Market_Cap"] = info.get("marketCap")

            # Calculate Yields
            if (
                row["Operating_Cash_Flow"]
                and row["Market_Cap"]
                and row["Market_Cap"] > 0
            ):
                row["OCF_Yield"] = row["Operating_Cash_Flow"] / row["Market_Cap"]
            else:
                row["OCF_Yield"] = None

            # Metadata
            row["YF_Sector"] = info.get("sector")
            row["YF_Industry"] = info.get("industry")
            row["Price"] = info.get("currentPrice")
            row["Currency_YF"] = info.get("currency")

            return row

        except Exception as e:
            str_e = str(e)

            # 404 is terminal, don't retry
            if "404" in str_e and "Not Found" in str_e:
                if debug:
                    print(f"[DEBUG] {yf_symbol}: 404 Not Found", file=sys.stderr)
                return None

            # Rate limits (429) or Server Errors (5xx)
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


def passes_filters(row, args):
    if not row:
        return False

    # Helper to safely parse floats
    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    # 1. Valuation: P/E check
    pe = safe_float(row.get("P/E"))
    if pe is None:
        if args.debug:
            print(f"[SKIP] {row.get('YF_Ticker')}: Missing P/E", file=sys.stderr)
        return False
    if pe > args.max_pe:
        if args.debug:
            print(
                f"[SKIP] {row.get('YF_Ticker')}: P/E {pe} > {args.max_pe}",
                file=sys.stderr,
            )
        return False

    # 2. Profitability: ROE or ROA check (OR logic between ROE/ROA, but AND with others)
    roe = safe_float(row.get("ROE"))
    roa = safe_float(row.get("ROA"))
    min_roe = args.min_roe / 100.0
    min_roa = args.min_roa / 100.0

    has_good_roe = roe is not None and roe > min_roe
    has_good_roa = roa is not None and roa > min_roa

    if not (has_good_roe or has_good_roa):
        if args.debug:
            print(
                f"[SKIP] {row.get('YF_Ticker')}: Low Profit (ROE={roe}, ROA={roa})",
                file=sys.stderr,
            )
        return False

    # 3. Health: Debt-to-Equity (Must be < Max D/E)
    de = safe_float(row.get("Debt_to_Equity"))
    if de is None:
        # Strict mode: Skip if D/E is unknown (often Financials)
        if args.debug:
            print(f"[SKIP] {row.get('YF_Ticker')}: Missing D/E", file=sys.stderr)
        return False
    if de > args.max_de:
        if args.debug:
            print(f"[SKIP] {row.get('YF_Ticker')}: High Debt ({de}%)", file=sys.stderr)
        return False

    # 4. Viability: Operating Cash Flow (Must be positive)
    ocf = safe_float(row.get("Operating_Cash_Flow"))
    if ocf is None or ocf <= 0:
        if args.debug:
            print(f"[SKIP] {row.get('YF_Ticker')}: Negative/No OCF", file=sys.stderr)
        return False

    if args.debug:
        print(
            f"[KEEP] {row.get('YF_Ticker')}: P/E={pe}, D/E={de}%, ROE={roe}",
            file=sys.stderr,
        )
    return True


def load_data_robust(filepath_or_buffer):
    """Robust CSV loader that handles bad lines and encoding issues."""
    try:
        # engine='python' and on_bad_lines='skip' helps with malformed rows (e.g. JSON dumps)
        df = pd.read_csv(filepath_or_buffer, on_bad_lines="skip", engine="python")

        # HEURISTIC: Fix Misaligned CSVs
        # If header has 8 cols but data has 9, pandas typically makes the Ticker the Index.
        # Check if Index looks like Tickers (short, dots/caps) and YF_Ticker looks like Company Names (long, spaces)
        if "YF_Ticker" in df.columns and len(df) > 0:
            sample_ticker_col = str(df["YF_Ticker"].iloc[0])
            sample_index = str(df.index[0])

            # If "YF_Ticker" column looks like "Royal Bank..." (space) AND index looks like "RY.TO" (dot/caps)
            if " " in sample_ticker_col and (
                "." in sample_index or sample_index.isupper()
            ):
                print(
                    "Warning: Detected CSV misalignment (Ticker in index). Fixing...",
                    file=sys.stderr,
                )
                df.reset_index(inplace=True)
                # Overwrite YF_Ticker with the index (the real ticker)
                # The first column after reset is usually 'index' or 'level_0'
                index_col = df.columns[0]
                df["YF_Ticker"] = df[index_col]

        return df
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Filter global tickers by P/E, ROE, Debt, and Cash Flow"
    )
    parser.add_argument(
        "--input", default="tickers.csv", help="Input CSV. Use '-' for stdin."
    )
    parser.add_argument("--output", default="filtered_gems.csv", help="Output CSV")

    # Filter Thresholds
    parser.add_argument("--max-pe", type=float, default=18.0, help="Max P/E Ratio")
    parser.add_argument("--min-roe", type=float, default=13.0, help="Min ROE %")
    parser.add_argument("--min-roa", type=float, default=6.0, help="Min ROA %")
    parser.add_argument(
        "--max-de", type=float, default=150.0, help="Max Debt/Equity % (150 = 1.5)"
    )

    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--debug", action="store_true", help="Show skip reasons")
    args = parser.parse_args()

    # 1. Load Data
    df_in = None
    if args.input == "-":
        if sys.stdin.isatty():
            print("Error: '--input -' specified but no data piped.", file=sys.stderr)
            sys.exit(1)
        print("Reading stdin...", file=sys.stderr)
        df_in = load_data_robust(sys.stdin)
    elif not os.path.exists(args.input):
        if not sys.stdin.isatty():
            print("Reading stdin (auto-detect)...", file=sys.stderr)
            df_in = load_data_robust(sys.stdin)
        else:
            print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        df_in = load_data_robust(args.input)

    if "YF_Ticker" not in df_in.columns:
        # Fallback: check if the first column looks like a ticker
        if "Ticker" in df_in.columns:
            df_in.rename(columns={"Ticker": "YF_Ticker"}, inplace=True)
        else:
            # Assume first column
            df_in.rename(columns={df_in.columns[0]: "YF_Ticker"}, inplace=True)
            print(
                "Warning: 'YF_Ticker' column not found, using first column as ticker.",
                file=sys.stderr,
            )

    records = df_in.to_dict("records")
    total = len(records)

    print(
        f"Scanning {total} tickers with {args.workers} workers... (Debug: {args.debug})",
        file=sys.stderr,
    )
    print(
        f"Criteria: P/E<{args.max_pe}, ROE>{args.min_roe}%/ROA>{args.min_roa}%, D/E<{args.max_de}%, OCF>0",
        file=sys.stderr,
    )

    results = []
    processed_count = 0
    start_time = time.time()

    write_header = not os.path.exists(args.output)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_row = {
                executor.submit(process_row, row, args.debug): row for row in records
            }

            for future in as_completed(future_to_row):
                processed_count += 1
                try:
                    data = future.result()

                    if passes_filters(data, args):
                        # Ensure all output columns exist (fill NaN if missing)
                        for col in OUTPUT_ORDER:
                            if col not in data:
                                data[col] = None
                        results.append(data)

                        # Incremental Save
                        if len(results) >= BATCH_SIZE:
                            df_batch = pd.DataFrame(results)
                            # Reorder columns
                            df_batch = df_batch.reindex(columns=OUTPUT_ORDER)
                            # Handle remaining columns if any (append them at end)
                            extra_cols = [
                                c for c in df_batch.columns if c not in OUTPUT_ORDER
                            ]
                            if extra_cols:
                                df_batch = pd.concat(
                                    [df_batch[OUTPUT_ORDER], df_batch[extra_cols]],
                                    axis=1,
                                )
                            else:
                                df_batch = df_batch[OUTPUT_ORDER]

                            df_batch.to_csv(
                                args.output, mode="a", header=write_header, index=False
                            )
                            write_header = False
                            results = []

                    if processed_count % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        print(
                            f"Progress: {processed_count}/{total} ({rate:.1f} t/s)",
                            file=sys.stderr,
                        )

                except Exception:
                    pass
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving pending results...", file=sys.stderr)

    if results:
        df_batch = pd.DataFrame(results)
        # Reorder columns logic (dup from above)
        available_cols = [c for c in OUTPUT_ORDER if c in df_batch.columns]
        df_batch = df_batch.reindex(
            columns=available_cols
            + [c for c in df_batch.columns if c not in available_cols]
        )
        df_batch.to_csv(args.output, mode="a", header=write_header, index=False)

    print(f"\nDone! Saved processed records to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
