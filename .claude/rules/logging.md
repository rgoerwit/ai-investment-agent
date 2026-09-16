---
paths:
  - "src/**/*.py"
---

# Logging standard

Every module under `src/` uses **structlog exclusively**.

```python
logger = structlog.get_logger(__name__)          # never logging.getLogger()
logger.info("article_generated", ticker=ticker)  # event name, then kwargs
```

- **Keyword-argument style only.** Never f-strings, `%`-formatting, or positional args
  after the event name — structlog's positional formatter raises `TypeError` on a
  placeholder-less event with positional args.
- **Event names are snake_case** (`article_generated`, not `"Article generated"`).
  Structured filtering and the log-parsing scripts depend on it.
- **Operator-visible exception logs** (warning and error) use
  `**summarize_exception(exc, operation=...)` from `src/error_safety.py`, never raw
  `error=str(exc)`. It redacts secrets and URLs and adds structured fields. Plain
  string context belongs under `reason=`, never `error=`. A debug-level raw `error=`
  is tolerated.
- **Content previews** at warning or error level must be wrapped in
  `redact_sensitive_text(...)`.
- **Error returns visible to a tool or an LLM are typed**, never raw exception prose.

Exceptions, for bootstrap and suppression reasons only: `src/config.py`,
`src/main.py`, `src/health_check.py`, `src/report_generator.py`.

Event names must stay **family-neutral** where a fallback chain can move between
vendors: put the provider and model in structured fields, not in the event name, or
the log lies after the first fallback.

`tests/test_logging_consistency.py` enforces all of this statically with AST checks.

Full history: `docs/CODEBASE_MEMORY.md`.
