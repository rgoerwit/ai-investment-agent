# Injection Payload Corpus Sources

This corpus is intentionally small, reviewed, and static. Entries are data only:
tests must never execute, import, shell-expand, fetch, or template these payloads
into commands.

## Current Sources

- `handcrafted@repo`: local cases targeting this repository's `SourceKind`
  boundaries and editor egress policy.

## Refresh Policy

- Do not auto-refresh this corpus from upstream projects.
- Future third-party samples must be vendored as JSON entries with pinned commit
  SHAs, source licenses, and human-reviewed diffs.
- Use `scripts/refresh_injection_corpus.py --source-file <path>` for a dry-run
  validation. Add `--write --source-sha <sha256>` only after reviewing the diff.
- Use `scripts/refresh_judge_replay.py --record` to re-record semantic judge
  replay responses. This intentionally calls the live judge model, requires a
  real `GOOGLE_API_KEY`, and should be reviewed before commit.
- External scanners and corpora such as promptfoo, garak, PyRIT, BIPIA, PINT,
  Open-Prompt-Injection, and deepset/prompt-injections must not be added as
  Poetry dependencies for this suite.
