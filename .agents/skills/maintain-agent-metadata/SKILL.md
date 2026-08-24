---
name: maintain-agent-metadata
description: Audit, extend, or reorganize this repository's Codex guidance and skills. Use for AGENTS.md, repository skill changes, metadata placement decisions, portability checks, or instruction-context reduction.
---

# Maintain public Codex metadata

This repository's Codex metadata is public and must work in a clean clone without any
user-level extension. It may not require another skill to exist.

## Place material by scope

- Keep only unconditional repository rules in `AGENTS.md`.
- Put a bounded procedure with a recognizable trigger in one focused
  `.agents/skills/<name>/SKILL.md`.
- Put durable architecture, rationale, or incident history in `docs/` without citing
  agent metadata.
- Move cross-repository preferences or workflows to the user-level layer, but never
  reference that destination from tracked metadata.
- Keep credentials, personal data, local setup, machine layout, and private notes out
  of every tracked destination.

## Audit every change

1. Resolve each referenced repository path. It must be tracked, supplied by a tracked
   template, or clearly described as output from normal setup or execution.
2. Reject ignored inputs, missing files, symlinks to outside content, absolute paths,
   home-relative paths, user-level skills, and local command wrappers.
3. `.env.example` is the only environment file whose contents may be read or quoted.
   Permit `.env` as a filename, but reject every other private `.env*` filename and all
   private environment-file contents, assignments, values, endpoints, identifiers, and
   holdings. Codex and metadata checks must not inspect or ingest those files;
   application runtime use is outside this metadata exception and must not expose
   values.
4. Keep other coding tools' metadata out of repository skills. A narrow no-touch
   boundary belongs in root guidance; no procedure may import or rely on that layer.
5. Keep skill names short and action-oriented. The frontmatter name must equal the
   directory name; the description must state the actual trigger clearly.
6. Prefer instructions over scripts unless deterministic repetition justifies code.
   Any supporting file must be tracked inside the same skill or elsewhere in the repo.
7. Keep root guidance below 12 KiB and avoid duplicating detail already owned by code,
   tests, or public documentation.

Run:

```bash
make docs-guards
make agent-metadata-guards
poetry run pytest tests/scripts/test_agent_metadata_guards.py -v
git diff --check
```

When auditing ignore behavior, use `git check-ignore -q AGENTS.md` and require a
nonzero result. Do not infer ignored status from `git check-ignore -v`: that diagnostic
form also reports a matching negation rule.

Finish with a manual scan of the complete public metadata surface for secret-like
strings, personal or holdings information, machine-specific paths, ignored-input
dependencies, and references that would be meaningless in a clean clone. Keep the
guard exemption count at zero.
