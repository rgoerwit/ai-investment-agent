"""Post-graph reporting utilities (memo rendering, source-confidence tables, etc.).

This package is the home for transformations applied to the analysis `result`
dict *after* the agent graph has completed. It lives separately from
`src/report_generator.py` to keep memo/table builders out of the large
ReportGenerator class and to make them straightforward to unit-test.
"""
