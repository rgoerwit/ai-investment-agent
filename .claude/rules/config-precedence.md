---
paths:
  - "src/config.py"
  - "src/*_config.py"
  - "src/runtime_config.py"
  - "src/ibkr/portfolio_defaults.py"
---

# Settings, defaults, and override precedence

## The canonical chain

```
CLI flag  >  shell environment variable  >  .env file  >  hardcoded default
```

Shell environment beats `.env` — that is pydantic-settings behaviour, and the
`SHELL ENVIRONMENT OVERRIDE DETECTED` warning is expected, not a bug.

## Where a default belongs

Each of these owns its defaults; **do not re-declare a literal that lives elsewhere.**

| Home | Owns |
|---|---|
| `src/config.py` `Settings` | env-driven application config; secrets via `SecretStr` read through `get_*()` accessors |
| `src/ibkr_config.py` | broker credentials and the portfolio knobs with an env override |
| `src/ibkr/portfolio_defaults.py` | **canonical literal defaults** for every portfolio knob; change a portfolio default *here only* |
| `src/runtime_config.py` `RuntimeConfig` | the run-scoped fields a CLI flag overrides |
| `src/llm_runtime/seats.py` `SeatExecutionPolicy` | per-seat call semantics: temperature, timeouts, retries, tier pins |

## Never write a fallback for a Settings field

**Read the field: `settings.api_timeout`, never `getattr(settings, "api_timeout", 300)`.**
A literal fallback copies a default that already lives in `Settings` and goes stale,
and for a gate it fails open: `getattr(config, "buy_stability_enabled", False)` and
`getattr(settings, "llm_require_review_independence", False)` were OFF for any config
object that lacked the field while `Settings` ships them ON. A config object that lacks
a field is a bug to surface (`AttributeError`), so tests build `Settings(_env_file=None,
...)` or `config.model_copy(update=...)`, not a partial `SimpleNamespace`.
`tests/config/test_settings_hygiene.py` scans `src/` for the shape.

**"Explicit settings, else the live singleton" has one spelling:**
`src.config.resolve_settings(cfg)`. It reads `src.config.config` at call time, so
tests patch that one name; never bind `config` at import as a fallback.

## Run-scoped values

Read them as `get_runtime_config(config).<field>` — **never mutate the global config
singleton for a CLI flag.** `RuntimeConfig` is ContextVar-scoped and task-local, so
concurrent runs in one process stay isolated. Add a field there only when a CLI flag
genuinely overrides it; everything else stays on `config`.

## Reading a key that has two spellings

When a setting exists under both a legacy and a current name, **read it through
exactly one function** and let that function decide. The failure mode is nasty: the
two spellings behave identically whenever the keys agree, so a divergence is invisible
until an operator sets only one of them — and then one half of the system acts on a
value the other half does not know about.

## Only add an override surface you actually consume

Dead config — a field or env var nothing reads — and re-literalized defaults are
guarded by `tests/config/test_settings_hygiene.py`. Note that the dead-config test
matches a field *name*, so it cannot catch a value that is accepted and then ignored;
that needs a behavioural test.

Assert shipped defaults via `Settings.model_fields[name].default`, never by
instantiating `Settings()` — instantiation loads the developer's `.env`, so a
documented operator override makes a correct configuration fail the test.

Full history: `docs/RUNTIME_MODEL.md`.
