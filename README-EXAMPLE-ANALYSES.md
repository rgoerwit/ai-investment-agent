# Example Analyses

This file explains the example analysis artifacts in `docs/examples/`. The
examples are useful, but they are not a contract for the current report format.

The report generator has changed over time. Current reports may include sections
that older examples do not have, such as:

- Investment Memo
- Analysis Validity
- Verification Caveats
- Red Flag Pre-Screening
- News & Catalysts
- External Consultant Review
- APAC Regional Specialist output
- PM_BLOCK-derived chart and verdict behavior

For the current report assembly code, read `src/report_generator.py`. For the
user-facing workflow, read `README.md`.

## What the Examples Are Good For

The examples show the kind of reasoning the system tries to produce:

- explicit thesis checks
- financial health and growth scoring
- liquidity and trading logistics
- data-quality warnings
- bull/bear synthesis
- risk sizing and final Portfolio Manager judgment

They are also useful as a reminder of the system's limitations. The reports can
be long. They can overstate weak signals. They can treat missing data too
conservatively or not conservatively enough. They can inherit errors from
providers. They are research artifacts, not decisions handed down from a more
reliable source.

## Current Report Shape, Roughly

A normal full report is usually assembled around these sections:

1. Thesis Compliance at a Glance
2. Thesis Alignment chart
3. Valuation chart
4. Investment Memo, when available
5. Executive Summary
6. Technical Analysis
7. Fundamental Analysis
8. Market Sentiment
9. News & Catalysts
10. Investment Recommendation, mainly as a fallback when PM output is missing
11. Consultant or regional review sections, when enabled
12. Trading Strategy, when the verdict calls for one
13. Risk Assessment

That list is intentionally approximate. Some sections are skipped in brief mode,
quick mode, failed/degraded runs, or when an optional agent is disabled.

## Example Reports in This Repo

The current checked-in examples live under:

- `docs/examples/analysis-reports/`
- `docs/examples/analysis-images/`

The filenames include the ticker and date. Treat them as dated examples of
system behavior, not as current truth about the companies.

## How to Read an Example

Read examples in this order:

1. Look at the final verdict and position size.
2. Check the hard-fail and thesis-compliance lines.
3. Read the DATA_BLOCK in Fundamental Analysis.
4. Look for data-quality warnings and consultant objections.
5. Compare the trading plan against the risk assessment.
6. Verify any important fact outside the report.

If the report has a BUY verdict but weak data quality, the right reaction is not
"the model found something." The right reaction is "this is a candidate for
manual verification."

## What Good Looks Like

A stronger report usually has:

- enough fundamental data to score the company without heroic assumptions
- liquidity above the minimum threshold
- valuation that is cheap for a reason you can understand
- growth or transition evidence from more than one source
- clear handling of ADR, PFIC, VIE, sanctions, and jurisdiction issues
- a consultant or auditor section that does not find a fatal factual problem
- a position size that reflects the actual risk, not the excitement of the story

## What Bad Looks Like

Weak reports often have:

- pervasive `N/A` values
- absurd provider metrics, such as impossible P/E or P/B values
- ticker, currency, or exchange confusion
- "undiscovered" arguments based only on lack of StockTwits chatter
- local-market claims that are not backed by filings or credible sources
- bullish conclusions that depend on one unverified catalyst
- trading plans that ignore liquidity or settlement realities

When you see these, reject the output or downgrade it to a lead for manual work.

## A Practical Workflow

For broad discovery:

1. Run the screening pipeline from `README.md`.
2. Review BUY names and near-misses.
3. Open the saved analysis JSON and markdown for each serious candidate.
4. Verify the core numbers against filings, company IR pages, and your broker.
5. Treat data-poor markets as data-poor markets. Do not let the system's prose
   make thin evidence feel thicker than it is.

This project can reduce the cost of first-pass research. It cannot remove the
need for judgment.
