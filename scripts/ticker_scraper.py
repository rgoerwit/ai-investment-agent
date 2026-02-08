#!/usr/bin/env python3
import argparse
import io
import json
import random
import sys
import time

import pandas as pd
import requests

# --- CONSTANTS & CONFIG ---
DEFAULT_CONFIG_PATH = "config/exchanges.json"
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
]

OUTPUT_COLUMNS = [
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


# --- SESSION ---
def get_session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
        }
    )
    return s


# --- CHECK DEPENDENCIES ---
def check_deps():
    import importlib.util

    missing = []
    for pkg in ("openpyxl", "xlrd", "lxml"):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)

    if missing:
        print("WARNING: Missing optional dependencies.", file=sys.stderr)
        print(f"Run: poetry add {' '.join(missing)}", file=sys.stderr)
        print(
            "Scraping will proceed but Excel/HTML sources will fail.\n", file=sys.stderr
        )


# --- LOADERS ---
def load_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)


def find_col_fuzzy(df, target):
    """Returns the actual column name in df that matches target (case-insensitive/stripped)."""
    if not target:
        return None
    target_clean = str(target).strip().lower()

    # 1. Exact match
    if target in df.columns:
        return target

    # 2. Case-insensitive match
    for col in df.columns:
        if str(col).strip().lower() == target_clean:
            return col

    return None


def standardize_dataframe(df, config):
    params = config.get("params", {})

    # Defensive renaming
    rename_dict = {}

    ticker_col = params.get("ticker_col")
    actual_ticker = find_col_fuzzy(df, ticker_col)
    if actual_ticker:
        rename_dict[actual_ticker] = "Ticker_Raw"

    name_col = params.get("name_col")
    actual_name = find_col_fuzzy(df, name_col)
    if actual_name:
        rename_dict[actual_name] = "Company"

    # Dynamic column mapping
    col_map = params.get("col_map", {})
    for std_col, source_col in col_map.items():
        actual_source = find_col_fuzzy(df, source_col)
        if actual_source:
            rename_dict[actual_source] = std_col

    df = df.rename(columns=rename_dict)

    # Add metadata
    df["Country"] = config["country"]
    df["Exchange"] = config["exchange_name"]

    # Fill missing standard columns
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df


def generate_yf_ticker(row, config):
    # Validation Hook
    if pd.isna(row.get("Ticker_Raw")) or str(row.get("Ticker_Raw")).strip() == "":
        return None

    raw = str(row["Ticker_Raw"]).strip()
    suffix = config["yahoo_suffix"]

    # Cleanup logic (e.g. pad HK tickers)
    if config.get("params", {}).get("clean_rule") == "pad_4_digits":
        try:
            raw = raw.split(".")[0].zfill(4)
        except Exception:
            pass

    # Dynamic Suffix Logic
    if suffix == "dynamic":
        suffix_map = config["params"].get("suffix_map", {})
        market_col = config["params"].get("market_col")

        # Try to match market column to suffix key
        if market_col and market_col in row and pd.notna(row[market_col]):
            market_val = str(row[market_col])
            for key, sfx in suffix_map.items():
                if key.lower() in market_val.lower():
                    return f"{raw}{sfx}"
        return raw  # Fallback

    return f"{raw}{suffix}"


# --- HANDLERS ---
def handle_download_json(config, session):
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


def handle_download_csv(config, session):
    response = session.get(config["source_url"])
    response.raise_for_status()

    params = config["params"]
    skip = params.get("skip_rows", 0)
    sep = params.get("delimiter", ",")

    # Use engine='python' for robustness against bad lines/buffer overflows
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
        except UnicodeDecodeError:
            continue
        except Exception:
            continue  # Try next encoding or fail

    raise ValueError("CSV parsing failed (check delimiter or encoding)")


def handle_download_excel(config, session):
    response = session.get(config["source_url"])
    response.raise_for_status()

    params = config["params"]
    sheet = params.get("sheet_name", 0)
    skip = params.get("skip_rows", 0)

    return pd.read_excel(io.BytesIO(response.content), sheet_name=sheet, skiprows=skip)


def handle_scrape_html(config, session):
    response = session.get(config["source_url"])
    response.raise_for_status()

    if "<table" not in response.text.lower():
        raise ValueError("No HTML tables found (likely JS-rendered page)")

    params = config["params"]
    idx = params.get("table_index", 0)

    # Use lxml flavor if available
    dfs = pd.read_html(io.StringIO(response.text), flavor=["lxml", "html5lib", "bs4"])

    if len(dfs) > idx:
        # Check if the primary choice actually has the column we want
        primary_df = dfs[idx]
        if find_col_fuzzy(primary_df, params.get("ticker_col")):
            return primary_df

    # Fallback: search ALL tables for the ticker column
    target_col = params.get("ticker_col")
    if target_col:
        for _i, df in enumerate(dfs):
            if find_col_fuzzy(df, target_col):
                return df

    raise ValueError(f"Table containing '{target_col}' not found")


# --- MAIN ---
def main():
    check_deps()
    parser = argparse.ArgumentParser()
    parser.add_argument("--configfile", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output")
    args = parser.parse_args()

    recipe = load_config(args.configfile)
    session = get_session()
    all_dfs = []

    print(f"Loaded {recipe['meta']['description']}", file=sys.stderr)

    for ex in recipe["exchanges"]:
        print(f"Processing {ex['country']}...", file=sys.stderr, end=" ")
        try:
            if ex["method"] == "download_csv":
                df = handle_download_csv(ex, session)
            elif ex["method"] == "download_excel":
                df = handle_download_excel(ex, session)
            elif ex["method"] == "scrape_html":
                df = handle_scrape_html(ex, session)
            elif ex["method"] == "download_json":
                df = handle_download_json(ex, session)
            else:
                print("Unknown method", file=sys.stderr)
                continue

            if df is None or df.empty:
                print("Empty DataFrame", file=sys.stderr)
                continue

            # Capture columns before standardization for debug
            raw_cols = list(df.columns)

            df = standardize_dataframe(df, ex)
            df["YF_Ticker"] = df.apply(
                lambda r, _ex=ex: generate_yf_ticker(r, _ex), axis=1
            )

            final_df = df[OUTPUT_COLUMNS].dropna(subset=["YF_Ticker"])
            all_dfs.append(final_df)

            if len(final_df) > 0:
                print(f"OK ({len(final_df)} rows)", file=sys.stderr)
            else:
                # Debug output for 0 rows
                print(
                    f"OK (0 rows) - Check Config. Found columns: {raw_cols}",
                    file=sys.stderr,
                )

            time.sleep(1.0)

        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)

    if all_dfs:
        master = pd.concat(all_dfs, ignore_index=True)
        master = master.drop_duplicates(subset=["YF_Ticker"])

        if args.output:
            master.to_csv(args.output, index=False)
            print(f"\nSaved {len(master)} rows to {args.output}", file=sys.stderr)
        else:
            master.to_csv(sys.stdout, index=False)
    else:
        print("No data extracted", file=sys.stderr)


if __name__ == "__main__":
    main()
