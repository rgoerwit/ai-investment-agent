import os
import time

import yfinance as yf

# Known CMIC / Sanctioned Keywords (Partial list for heuristic screening)
CMIC_KEYWORDS = [
    "CNOOC",
    "PETROCHINA",
    "CHINA MOBILE",
    "CHINA TELECOM",
    "CHINA UNICOM",
    "SMIC",
    "SEMICONDUCTOR MANUFACTURING INTERNATIONAL",
    "HIKVISION",
    "DJI",
    "SENSETIME",
    "AVIC",
    "NORINCO",
    "CSSC",
    "CHINA RAILWAY CONSTRUCTION",
    "CRRC",
    "CHINA STATE SHIPBUILDING",
    "CHINA AEROSPACE",
    "CHINA COMMUNICATIONS CONSTRUCTION",
    "CHINA NUCLEAR",
    "CHINA GENERAL NUCLEAR",
]

# PFIC Flags (Industry/Name)
PFIC_KEYWORDS = [
    "ASSET MANAGEMENT",
    "INVESTMENT TRUST",
    "CLOSED END FUND",
    "CAPITAL",
    "HOLDINGS",
]

# VIE Risky Sectors (China)
VIE_SECTORS = ["Technology", "Communication Services", "Consumer Cyclical"]


def check_ticker(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info

        name = info.get("longName", "").upper()
        country = info.get("country", "Unknown")
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        flags = []

        # 1. CMIC / Sanctions
        for kw in CMIC_KEYWORDS:
            if kw in name:
                flags.append(f"CMIC_MATCH({kw})")

        # 2. Country Risk
        if country in ["China", "Hong Kong", "Russia"]:
            # VIE Check
            if country == "China" or country == "Hong Kong":
                if sector in VIE_SECTORS:
                    flags.append("VIE_RISK_SECTOR")

            if country == "Russia":
                flags.append("RUSSIA_SANCTIONS")

        # 3. PFIC Check (Heuristic)
        # Non-US + Financial/Asset Mgmt keywords
        if country != "United States":
            for kw in PFIC_KEYWORDS:
                if kw in name:
                    # Refine: Some "Holdings" are operating companies.
                    # If sector is Financial Services, higher risk.
                    if sector == "Financial Services":
                        flags.append(f"PFIC_RISK({kw})")
                    elif (
                        kw != "HOLDINGS"
                    ):  # Investment trusts etc are almost always PFIC
                        flags.append(f"PFIC_RISK({kw})")

        return {
            "ticker": ticker,
            "name": name,
            "country": country,
            "sector": sector,
            "industry": industry,
            "flags": flags,
            "clean": len(flags) == 0,
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e),
            "clean": False,  # Assume dirty if we can't verify
        }


def main():
    source_file = (
        "scratch/ticker_filter_results_exus.txt"  # Use the unchecked list we just made
    )
    output_file = "scratch/ticker_filter_results-CLEAN.txt"
    log_file = "scratch/ticker_filter_results-LOG.txt"

    if not os.path.exists(source_file):
        # Fallback to the original if unchecked doesn't exist
        source_file = "scratch/sample_tickers.txt"

    print(f"Reading from {source_file}...")

    try:
        with open(source_file) as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("No ticker file found.")
        return

    print(f"Screening {len(tickers)} tickers. This may take a moment (rate limits)...")

    clean_tickers = []
    log_entries = []

    # Process in chunks to avoid overwhelming yfinance/network
    # Increased limit to 200 based on initial success

    count = 0
    # simple skip logic if we want to resume?
    # For now just start from 0 but limit to 200

    start_index = 0  # Resume after the first 250

    print(f"Resuming from index {start_index}...")

    current_batch = tickers[start_index:]

    for ticker in current_batch:
        # Rate limit safety
        if count > 0 and count % 10 == 0:
            time.sleep(0.5)  # Reduced sleep slightly

        result = check_ticker(ticker)

        if result.get("clean"):
            clean_tickers.append(ticker)
            print(f"[CLEAN] {ticker} - {result.get('name')}")
        else:
            reason = ", ".join(result.get("flags", [result.get("error", "Unknown")]))
            print(f"[FLAGGED] {ticker} - {result.get('name')}: {reason}")
            log_entries.append(f"{ticker}: {reason}")

        count += 1
        # Hard limit for this interaction
        if count >= 400:
            print("Stopping after 400 tickers (batch limit).")
            break

    # Append mode
    with open(output_file, "a") as f:
        for t in clean_tickers:
            f.write(t + "\n")

    with open(log_file, "a") as f:
        for entry in log_entries:
            f.write(entry + "\n")

    print(
        f"\nScreening complete (partial). {len(clean_tickers)} clean tickers found out of {count} checked."
    )
    print(f"Clean list written to {output_file}")
    print(f"Flags logged to {log_file}")


if __name__ == "__main__":
    main()
